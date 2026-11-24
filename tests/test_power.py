"""The power script: status, stop and start of what bills, against fake AWS clients."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aws_power as power  # noqa: E402

CLUSTER_ARN = "arn:aws:ecs:ap-southeast-2:1:cluster/sepsis-dev-training-cluster"
SERVICE_ARN = "arn:aws:ecs:ap-southeast-2:1:service/sepsis-dev-training-cluster/sepsis-dev-mlflow"
UI_SERVICE_ARN = "arn:aws:ecs:ap-southeast-2:1:service/sepsis-dev-training-cluster/sepsis-dev-evidently"


class World:
    """One shared state behind the four fake clients."""

    def __init__(self, db_status="available", running=1, one_off=False):
        self.db_status = db_status
        self.services = {
            "sepsis-dev-mlflow": {"desired": running, "running": running, "group": "mlflow", "wanted": "1"},
            "sepsis-dev-evidently": {"desired": running, "running": running, "group": "evidently", "wanted": "1"},
        }
        self.one_off = one_off
        self.rules = []            # (group id, cidr)
        self.calls = []
        self.sleeps = 0

    def session(self):
        world = self

        class Cf:
            def list_stack_resources(self, StackName, NextToken=None):
                items = [("AWS::ECS::Cluster", "Cluster", "sepsis-dev-training-cluster"),
                         ("AWS::ECS::Service", "MlflowService", SERVICE_ARN),
                         ("AWS::ECS::Service", "EvidentlyService", UI_SERVICE_ARN),
                         ("AWS::RDS::DBInstance", "Database", "db-1"),
                         ("AWS::EC2::SecurityGroup", "MlflowSecurityGroup", "sg-mlflow"),
                         ("AWS::EC2::SecurityGroup", "TaskSecurityGroup", "sg-tasks"),
                         ("AWS::EC2::SecurityGroup", "UiSecurityGroup", "sg-ui"),
                         ("AWS::ElasticLoadBalancingV2::LoadBalancer", "Alb", "arn:alb")]
                return {"StackResourceSummaries": [{"ResourceType": t, "LogicalResourceId": l, "PhysicalResourceId": p,
                                                    "ResourceStatus": "CREATE_COMPLETE"} for t, l, p in items]}

        class Ecs:
            def list_tags_for_resource(self, resourceArn):
                name = resourceArn.rsplit("/", 1)[-1]
                svc = world.services[name]
                return {"tags": [{"key": "power:group", "value": svc["group"]}, {"key": "power:desired", "value": svc["wanted"]}]}

            def describe_services(self, cluster, services):
                svc = world.services[services[0]]
                group = "sg-mlflow" if services[0] == "sepsis-dev-mlflow" else "sg-ui"
                return {"services": [{"desiredCount": svc["desired"], "runningCount": svc["running"], "taskDefinition": "td:1",
                                      "networkConfiguration": {"awsvpcConfiguration": {"securityGroups": [group]}}}]}

            def describe_task_definition(self, taskDefinition):
                return {"taskDefinition": {"cpu": "512", "memory": "2048"}}

            def update_service(self, cluster, service, desiredCount):
                world.calls.append(("update_service", service, desiredCount))
                world.services[service]["desired"] = desiredCount
                world.services[service]["running"] = desiredCount

            def list_tasks(self, cluster, desiredStatus=None, serviceName=None):
                if serviceName:
                    return {"taskArns": ["t-svc"] if world.services[serviceName]["running"] else []}
                return {"taskArns": ["t-one", "t-svc"] if world.one_off else ["t-svc"]}

            def describe_tasks(self, cluster, tasks):
                out = []
                for task in tasks:
                    if task == "t-one":
                        out.append({"taskArn": "t-one", "group": "family:sepsis-dev-training"})
                    else:
                        out.append({"taskArn": task, "group": "service:sepsis-dev",
                                    "attachments": [{"details": [{"name": "networkInterfaceId", "value": "eni-1"}]}]})
                return {"tasks": out}

        class Rds:
            def describe_db_instances(self, DBInstanceIdentifier):
                return {"DBInstances": [{"DBInstanceStatus": world.db_status, "DBInstanceClass": "db.t4g.micro"}]}

            def stop_db_instance(self, DBInstanceIdentifier):
                world.calls.append(("stop_db", DBInstanceIdentifier))
                world.db_status = "stopping"

            def start_db_instance(self, DBInstanceIdentifier):
                world.calls.append(("start_db", DBInstanceIdentifier))
                world.db_status = "starting"

        class Ec2:
            def describe_security_groups(self, GroupIds):
                def rules(group, port):
                    return [{"IpProtocol": "tcp", "FromPort": port, "ToPort": port,
                             "IpRanges": [{"CidrIp": cidr, "Description": f"power-allow {cidr}"}]} for g, cidr in world.rules if g == group]

                return {"SecurityGroups": [
                    {"GroupId": "sg-mlflow", "Tags": [{"Key": "power:ui-port", "Value": "5000"}], "IpPermissions": rules("sg-mlflow", 5000)},
                    {"GroupId": "sg-ui", "Tags": [{"Key": "power:ui-port", "Value": "8000"}], "IpPermissions": rules("sg-ui", 8000)},
                    {"GroupId": "sg-tasks", "Tags": [], "IpPermissions": []}]}

            def authorize_security_group_ingress(self, GroupId, IpPermissions):
                world.calls.append(("authorize", GroupId, IpPermissions[0]["FromPort"]))
                cidr = IpPermissions[0]["IpRanges"][0]["CidrIp"]
                if (GroupId, cidr) in world.rules:
                    raise RuntimeError("InvalidPermission.Duplicate")
                world.rules.append((GroupId, cidr))

            def revoke_security_group_ingress(self, GroupId, IpPermissions):
                world.rules.remove((GroupId, IpPermissions[0]["IpRanges"][0]["CidrIp"]))
                world.calls.append(("revoke", GroupId))

            def describe_network_interfaces(self, NetworkInterfaceIds):
                return {"NetworkInterfaces": [{"Association": {"PublicIp": "54.1.2.3"}}]}

        clients = {"ecs": Ecs(), "rds": Rds(), "ec2": Ec2(), "cloudformation": Cf()}
        return type("Session", (), {"client": staticmethod(lambda name: clients[name])})()

    def power(self):
        def fake_sleep(_):
            self.sleeps += 1
            if self.db_status in ("starting", "stopping"):
                self.db_status = "available" if self.db_status == "starting" else "stopped"
        return power.Power(self.session(), "sepsis-dev", sleep=fake_sleep, timeout=100000)


def states(result):
    return {component: status for status, component, _ in result.rows}


class PowerTests(unittest.TestCase):
    def test_status_shows_what_is_running_and_the_estimated_cost(self):
        result = World().power().status()
        table = result.table()
        self.assertIn("sepsis-dev-mlflow", table)
        self.assertIn("running: 1/1", table)
        self.assertIn("RDS db-1", table)
        self.assertIn("Load balancer Alb", table)                       # reported, cannot be stopped
        cost = next(detail for _, c, detail in result.rows if c == "Estimated running cost")
        self.assertIn("/hour now", cost)
        self.assertGreater(float(cost.split("$")[1].split("/")[0]), 0)

    def test_status_of_a_stopped_environment_costs_nothing_but_storage(self):
        world = World(db_status="stopped", running=0)
        result = world.power().status()
        cost = next(detail for _, c, detail in result.rows if c == "Estimated running cost")
        self.assertTrue(cost.startswith("about $0.00/hour"))
        self.assertIn("storage only", result.table())

    def test_stop_scales_the_services_to_zero_stops_the_database_and_closes_the_browser_rules(self):
        world = World()
        world.rules.append(("sg-mlflow", "203.0.113.7/32"))
        result = world.power().stop()
        self.assertFalse(result.failed())
        self.assertEqual({s["desired"] for s in world.services.values()}, {0})
        self.assertIn(("stop_db", "db-1"), world.calls)
        self.assertEqual(world.rules, [])

    def test_stopping_only_the_mlflow_server_leaves_the_database_and_the_other_tool_alone(self):
        world = World()
        world.power().stop("mlflow")
        self.assertEqual(world.services["sepsis-dev-mlflow"]["desired"], 0)
        self.assertEqual(world.services["sepsis-dev-evidently"]["desired"], 1)
        self.assertNotIn(("stop_db", "db-1"), world.calls)

    def test_a_running_pipeline_step_blocks_stop_unless_forced(self):
        world = World(one_off=True)
        result = world.power().stop()
        self.assertTrue(result.failed())
        self.assertEqual(world.services["sepsis-dev-mlflow"]["desired"], 1)
        self.assertFalse(World(one_off=True).power().stop(force=True).failed())

    def test_stop_twice_changes_nothing_the_second_time(self):
        world = World(db_status="stopped", running=0)
        world.power().stop()
        self.assertEqual(world.calls, [])

    def test_start_brings_up_the_database_first_then_restores_the_service_counts_from_the_tags(self):
        world = World(db_status="stopped", running=0)
        result = world.power().start()
        self.assertFalse(result.failed())
        self.assertEqual(world.calls[0], ("start_db", "db-1"))
        self.assertEqual({s["desired"] for s in world.services.values()}, {1})
        self.assertGreater(world.sleeps, 0)                              # it waited for the database

    def test_starting_the_mlflow_server_starts_its_database_but_not_the_other_tool(self):
        world = World(db_status="stopped", running=0)
        world.power().start("mlflow")
        self.assertIn(("start_db", "db-1"), world.calls)
        self.assertEqual(world.services["sepsis-dev-mlflow"]["desired"], 1)
        self.assertEqual(world.services["sepsis-dev-evidently"]["desired"], 0)

    def test_start_can_open_the_ui_to_one_address_and_prints_the_url(self):
        world = World()
        result = world.power().start("mlflow", allow_cidr="203.0.113.7/32")
        self.assertEqual(world.rules, [("sg-mlflow", "203.0.113.7/32")])
        self.assertIn("http://54.1.2.3:5000", result.table())
        world.power().start("mlflow", allow_cidr="203.0.113.7/32")      # opening twice is not an error
        self.assertEqual(len(world.rules), 1)

    def test_each_tool_gets_its_own_port_and_url(self):
        world = World()
        result = world.power().start(allow_cidr="203.0.113.7/32")
        self.assertIn(("sg-mlflow", "203.0.113.7/32"), world.rules)
        self.assertIn(("sg-ui", "203.0.113.7/32"), world.rules)
        table = result.table()
        self.assertIn("mlflow: http://54.1.2.3:5000", table)
        self.assertIn("evidently: http://54.1.2.3:8000", table)
        self.assertIn(("authorize", "sg-ui", 8000), world.calls)
        world.power().stop()
        self.assertEqual(world.rules, [])                        # stop closes every rule it opened, on both tools

    def test_the_whole_internet_needs_an_explicit_flag(self):
        with self.assertRaisesRegex(ValueError, "whole internet"):
            power.check_cidr("0.0.0.0/0", False)
        self.assertEqual(power.check_cidr("0.0.0.0/0", True), "0.0.0.0/0")
        self.assertEqual(power.check_cidr("203.0.113.7", False), "203.0.113.7/32")
        with self.assertRaisesRegex(ValueError, "IPv4"):
            power.check_cidr("2001:db8::/32", False)
        with self.assertRaises(ValueError):
            power.check_cidr("not an address", False)
        with self.assertRaises(ValueError):
            World().power().start("all", allow_cidr="0.0.0.0/0")

    def test_unknown_components_are_refused(self):
        with self.assertRaisesRegex(ValueError, "component must be"):
            power.wanted("everything-else")


class WorkflowAndScriptTests(unittest.TestCase):
    def test_the_power_workflow_is_manual_with_no_schedule_and_no_inline_inputs(self):
        import yaml

        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "power.yml").read_text(encoding="utf-8"))
        self.assertEqual(set(workflow[True]), {"workflow_dispatch"})           # no schedule, no push trigger
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["action"]["options"], ["status", "start", "stop"])
        self.assertEqual(inputs["component"]["options"], ["all", "mlflow", "evidently", "database"])
        self.assertEqual(workflow["permissions"]["id-token"], "write")
        for step in workflow["jobs"]["power"]["steps"]:
            self.assertNotIn("${{ inputs.", step.get("run", ""), step.get("name"))   # inputs reach the script through env only

    def test_the_deploy_script_exposes_power_and_the_pipeline_needs_the_tools_on(self):
        text = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")
        self.assertIn("power) shift; cmd_power", text)
        self.assertIn("cmd_power require-on --component mlflow", text)

    def test_require_on_fails_while_the_tools_are_stopped(self):
        stopped = World(db_status="stopped", running=0).power().require_on("mlflow")
        self.assertTrue(stopped.failed())
        self.assertIn("run the power workflow with action start first", stopped.table())
        self.assertFalse(World().power().require_on("mlflow").failed())


if __name__ == "__main__":
    unittest.main()
