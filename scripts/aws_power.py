#!/usr/bin/env python3
"""Show, stop and start everything in the stack that bills while it is on.

    python scripts/aws_power.py --stack sepsis-dev --region ap-southeast-2 status
    python scripts/aws_power.py --stack sepsis-dev --region ap-southeast-2 start --allow-cidr 203.0.113.7/32
    python scripts/aws_power.py --stack sepsis-dev --region ap-southeast-2 stop

What it manages (found from the stack's own resources, so it needs no configuration):

* ECS services tagged ``power:group`` (the MLflow server, the Evidently UI): stop sets the desired count to 0,
  start restores the count in the service's ``power:desired`` tag (so the template stays the one place that says
  how many should run).
* RDS databases tagged ``power:group=database``: stop and start. AWS starts a stopped database again after
  7 days, so run ``status`` or ``stop`` again if a demo environment sits idle for a week.
* Browser access: ``start --allow-cidr`` opens the UI ports of the tools to that address only, and ``stop`` closes
  every rule it opened. 0.0.0.0/0 needs ``--allow-anywhere``.

Reported but not changed (they cannot be stopped, only removed): load balancers, NAT gateways and Elastic IPs.
One-off tasks (a running pipeline step) block ``stop`` unless ``--force`` is given.

No schedule and no automation: it runs when a person runs it (the **power** workflow or this script).
"""

import argparse
import ipaddress
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aws_verify import Result, of_type, resources  # noqa: E402

GROUPS = ("mlflow", "evidently", "database")
RULE_PREFIX = "power-allow"
# Rough Sydney prices (USD per hour), for the "about" line only; the AWS pricing page is the authority.
VCPU_HOUR, GB_HOUR, PUBLIC_IP_HOUR = 0.05, 0.0055, 0.005
DB_HOUR = {"db.t4g.micro": 0.02, "db.t4g.small": 0.04, "db.t3.micro": 0.02}


def service_parts(arn: str) -> tuple[str, str]:
    """(cluster name, service name) from a service ARN or a ``cluster/service`` string."""
    tail = arn.split("service/", 1)[-1]
    cluster, _, name = tail.partition("/")
    return cluster, name


def group_of(tags: dict) -> str | None:
    return tags.get("power:group")


def wanted(component: str) -> set[str]:
    if component == "all":
        return set(GROUPS)
    if component not in GROUPS:
        raise ValueError(f"component must be all, {', '.join(GROUPS)}")
    # The MLflow server and the Evidently UI cannot work without the database, so starting them starts it too.
    return {component} | ({"database"} if component == "mlflow" else set())


class Power:
    def __init__(self, session, stack: str, sleep=time.sleep, timeout: int = 900):
        self.ecs, self.rds, self.ec2 = session.client("ecs"), session.client("rds"), session.client("ec2")
        self.cloudformation = session.client("cloudformation")
        self.stack, self.sleep, self.timeout = stack, sleep, timeout
        self.items = resources(self.cloudformation, stack)
        self.result = Result()

    # ---- discovery
    def services(self) -> list[dict]:
        found = []
        for item in of_type(self.items, "AWS::ECS::Service"):
            cluster, name = service_parts(item["physical"])
            arn = item["physical"]
            tags = {t["key"]: t["value"] for t in self.ecs.list_tags_for_resource(resourceArn=arn).get("tags", [])}
            described = self.ecs.describe_services(cluster=cluster, services=[name])["services"]
            if not described:
                continue
            service = described[0]
            found.append({"cluster": cluster, "name": name, "arn": arn, "group": group_of(tags),
                          "desired_tag": int(tags.get("power:desired", "1")), "desired": service["desiredCount"],
                          "running": service["runningCount"], "task_definition": service["taskDefinition"],
                          "security_groups": service.get("networkConfiguration", {}).get("awsvpcConfiguration", {}).get("securityGroups", [])})
        return found

    def databases(self) -> list[dict]:
        found = []
        for item in of_type(self.items, "AWS::RDS::DBInstance"):
            described = self.rds.describe_db_instances(DBInstanceIdentifier=item["physical"])["DBInstances"]
            if described:
                db = described[0]
                found.append({"id": item["physical"], "status": db["DBInstanceStatus"], "class": db.get("DBInstanceClass", "")})
        return found

    def one_off_tasks(self) -> list[str]:
        """Running tasks that no service owns: pipeline steps, the approval task."""
        names = []
        for item in of_type(self.items, "AWS::ECS::Cluster"):
            listed = self.ecs.list_tasks(cluster=item["physical"], desiredStatus="RUNNING").get("taskArns", [])
            if not listed:
                continue
            for task in self.ecs.describe_tasks(cluster=item["physical"], tasks=listed)["tasks"]:
                if not str(task.get("group", "")).startswith("service:"):
                    names.append(f"{item['physical']}: {task.get('group', task['taskArn'])}")
        return names

    def cost_per_hour(self, service: dict) -> float:
        definition = self.ecs.describe_task_definition(taskDefinition=service["task_definition"])["taskDefinition"]
        vcpu, memory_gb = int(definition["cpu"]) / 1024, int(definition["memory"]) / 1024
        return service["running"] * (vcpu * VCPU_HOUR + memory_gb * GB_HOUR + PUBLIC_IP_HOUR)

    # ---- status
    def status(self) -> Result:
        burn = 0.0
        for service in self.services():
            state = "running" if service["running"] else ("stopped" if service["desired"] == 0 else "starting")
            cost = self.cost_per_hour(service)
            burn += cost
            self.result.add("WARN" if service["running"] else "PASS",
                            f"ECS service {service['name']} ({service['group'] or 'not managed'})",
                            f"{state}: {service['running']}/{service['desired']} tasks" + (f", about ${cost:.3f}/hour" if cost else ""))
        for db in self.databases():
            on = db["status"] in ("available", "starting", "configuring-enhanced-monitoring")
            cost = DB_HOUR.get(db["class"], 0.03) if on else 0.0
            burn += cost
            note = f", about ${cost:.3f}/hour plus storage" if on else ", storage only"
            if db["status"] == "stopped":
                note += " (AWS restarts a stopped database after 7 days)"
            self.result.add("WARN" if on else "PASS", f"RDS {db['id']}", f"{db['status']}{note}")
        tasks = self.one_off_tasks()
        self.result.add("WARN" if tasks else "PASS", "One-off tasks (pipeline steps)", "; ".join(tasks) if tasks else "none running")
        for kind, label in (("AWS::ElasticLoadBalancingV2::LoadBalancer", "Load balancer"),
                            ("AWS::EC2::NatGateway", "NAT gateway"), ("AWS::EC2::EIP", "Elastic IP")):
            for item in of_type(self.items, kind):
                self.result.add("WARN", f"{label} {item['logical']}", "bills every hour and cannot be stopped; remove it by changing the stack")
        self.result.add("PASS" if burn == 0 else "WARN", "Estimated running cost",
                        f"about ${burn:.2f}/hour now (${burn * 24:.2f}/day); storage, KMS and logs are not included")
        return self.result

    # ---- precondition for work that needs the tools (the pipeline)
    def require_on(self, component: str = "mlflow") -> Result:
        groups = wanted(component)
        for service in self.services():
            if service["group"] in groups:
                ok = service["running"] >= 1
                self.result.add("PASS" if ok else "FAIL", f"ECS service {service['name']}",
                                "running" if ok else "stopped: run the power workflow with action start first")
        if "database" in groups:
            for db in self.databases():
                ok = db["status"] == "available"
                self.result.add("PASS" if ok else "FAIL", f"RDS {db['id']}",
                                "available" if ok else f"{db['status']}: run the power workflow with action start first")
        return self.result

    # ---- stop
    def stop(self, component: str = "all", force: bool = False) -> Result:
        groups = wanted(component) - ({"database"} if component == "mlflow" else set())
        running = self.one_off_tasks()
        if running and not force:
            self.result.add("FAIL", "One-off tasks (pipeline steps)", f"still running: {'; '.join(running)}. Wait, or stop with force")
            return self.result
        for service in self.services():
            if service["group"] in groups:
                if service["desired"] != 0:
                    self.ecs.update_service(cluster=service["cluster"], service=service["name"], desiredCount=0)
                self.result.add("PASS", f"ECS service {service['name']}", "stopped (desired count 0)")
        for rule in self.close_access():
            self.result.add("PASS", "Browser access", f"closed {rule}")
        if "database" in groups:
            for db in self.databases():
                if db["status"] == "available":
                    self.rds.stop_db_instance(DBInstanceIdentifier=db["id"])
                    self.result.add("PASS", f"RDS {db['id']}", "stopping (storage is still billed)")
                else:
                    self.result.add("PASS", f"RDS {db['id']}", f"already {db['status']}")
        return self.result

    # ---- start
    def start(self, component: str = "all", allow_cidr: str | None = None, allow_anywhere: bool = False) -> Result:
        groups = wanted(component)
        if allow_cidr:
            allow_cidr = check_cidr(allow_cidr, allow_anywhere)
        if "database" in groups:
            for db in self.databases():
                if db["status"] == "stopped":
                    self.rds.start_db_instance(DBInstanceIdentifier=db["id"])
                if db["status"] != "available":
                    self.wait(lambda db=db: self.rds.describe_db_instances(DBInstanceIdentifier=db["id"])["DBInstances"][0]["DBInstanceStatus"] == "available",
                              f"database {db['id']}")
                self.result.add("PASS", f"RDS {db['id']}", "available")
        for service in self.services():
            if service["group"] in groups:
                want = service["desired_tag"]
                if service["desired"] != want:
                    self.ecs.update_service(cluster=service["cluster"], service=service["name"], desiredCount=want)
                self.wait(lambda s=service, want=want: self.ecs.describe_services(cluster=s["cluster"], services=[s["name"]])["services"][0]["runningCount"] >= want,
                          f"service {service['name']}")
                self.result.add("PASS", f"ECS service {service['name']}", f"running {want} task(s)")
        if allow_cidr:
            started = {sg for service in self.services() if service["group"] in groups for sg in service["security_groups"]}
            for line in self.open_access(allow_cidr, started):
                self.result.add("PASS", "Browser access", line)
        for line in self.urls():
            self.result.add("PASS", "Open in a browser", line)
        return self.result

    def wait(self, ready, label: str) -> None:
        deadline = time.monotonic() + self.timeout
        while not ready():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{label} did not become ready within {self.timeout} seconds")
            self.sleep(15)

    # ---- browser access
    def ui_groups(self) -> list[dict]:
        """Security groups of the tools, with the port each one serves its UI on (tag power:ui-port)."""
        ids = [i["physical"] for i in of_type(self.items, "AWS::EC2::SecurityGroup") if i["physical"]]
        if not ids:
            return []
        groups = self.ec2.describe_security_groups(GroupIds=ids)["SecurityGroups"]
        found = []
        for group in groups:
            tags = {t["Key"]: t["Value"] for t in group.get("Tags", [])}
            if "power:ui-port" in tags:
                found.append({"id": group["GroupId"], "port": int(tags["power:ui-port"]),
                              "rules": group.get("IpPermissions", [])})
        return found

    def open_access(self, cidr: str, only: set | None = None) -> list[str]:
        lines = []
        for group in self.ui_groups():
            if only is not None and group["id"] not in only:
                continue
            try:
                self.ec2.authorize_security_group_ingress(GroupId=group["id"], IpPermissions=[{
                    "IpProtocol": "tcp", "FromPort": group["port"], "ToPort": group["port"],
                    "IpRanges": [{"CidrIp": cidr, "Description": f"{RULE_PREFIX} {cidr}"}]}])
            except Exception as error:  # noqa: BLE001
                if "Duplicate" not in str(error):
                    raise
            lines.append(f"port {group['port']} open to {cidr} on {group['id']}")
        return lines

    def close_access(self) -> list[str]:
        closed = []
        for group in self.ui_groups():
            for permission in group["rules"]:
                for ip_range in permission.get("IpRanges", []):
                    if str(ip_range.get("Description", "")).startswith(RULE_PREFIX):
                        self.ec2.revoke_security_group_ingress(GroupId=group["id"], IpPermissions=[{
                            "IpProtocol": permission["IpProtocol"], "FromPort": permission["FromPort"],
                            "ToPort": permission["ToPort"], "IpRanges": [{"CidrIp": ip_range["CidrIp"]}]}])
                        closed.append(f"port {permission['FromPort']} for {ip_range['CidrIp']} on {group['id']}")
        return closed

    def urls(self) -> list[str]:
        lines = []
        ports = {g["id"]: g["port"] for g in self.ui_groups()}
        for service in self.services():
            service_ports = [ports[g] for g in service["security_groups"] if g in ports]
            if service["group"] not in ("mlflow", "evidently") or not service["running"]:
                continue
            listed = self.ecs.list_tasks(cluster=service["cluster"], serviceName=service["name"], desiredStatus="RUNNING").get("taskArns", [])
            if not listed:
                continue
            for task in self.ecs.describe_tasks(cluster=service["cluster"], tasks=listed)["tasks"]:
                for attachment in task.get("attachments", []):
                    details = {d["name"]: d["value"] for d in attachment.get("details", [])}
                    eni = details.get("networkInterfaceId")
                    if not eni:
                        continue
                    association = self.ec2.describe_network_interfaces(NetworkInterfaceIds=[eni])["NetworkInterfaces"][0].get("Association", {})
                    if association.get("PublicIp"):
                        port = service_ports[0] if service_ports else (5000 if service["group"] == "mlflow" else 8000)
                        lines.append(f"{service['group']}: http://{association['PublicIp']}:{port}")
        return lines


def check_cidr(value: str, allow_anywhere: bool) -> str:
    network = ipaddress.ip_network(value.strip(), strict=False)
    if network.version != 4:
        raise ValueError("only IPv4 addresses are supported")
    if network.prefixlen == 0 and not allow_anywhere:
        raise ValueError("0.0.0.0/0 opens the tools to the whole internet; use your own address (x.x.x.x/32), or pass --allow-anywhere")
    return str(network)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["status", "start", "stop", "require-on"])
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--component", default="all", help="all, mlflow, evidently or database")
    parser.add_argument("--allow-cidr", default="", help="start: open the tool UIs to this address, for example 203.0.113.7/32")
    parser.add_argument("--allow-anywhere", action="store_true", help="allow --allow-cidr 0.0.0.0/0")
    parser.add_argument("--force", action="store_true", help="stop: also when a pipeline step is running")
    args = parser.parse_args(argv)
    import boto3

    power = Power(boto3.Session(region_name=args.region), args.stack)
    try:
        if args.action == "status":
            result = power.status()
        elif args.action == "require-on":
            result = power.require_on(args.component)
        elif args.action == "stop":
            result = power.stop(args.component, args.force)
        else:
            result = power.start(args.component, args.allow_cidr or None, args.allow_anywhere)
    except (ValueError, TimeoutError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(result.table())
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(result.markdown().replace("### Deployment check", f"### Power {args.action}"))
    return 1 if result.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
