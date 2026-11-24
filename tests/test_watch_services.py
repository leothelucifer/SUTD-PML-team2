"""The watcher prints why a task stopped, cancels a crash-looping update, and gates the deploy on healthy services."""

import unittest
from datetime import UTC, datetime, timedelta

from scripts import aws_watch_services as watcher

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


class FakeEcs:
    def __init__(self, services, stopped=(), running=()):
        self.services, self.stopped, self.running = services, list(stopped), list(running)

    def describe_services(self, cluster, services):
        return {"services": [s for s in self.services if s["serviceName"] in services]}

    def list_tasks(self, cluster, serviceName, desiredStatus):
        source = self.stopped if desiredStatus == "STOPPED" else self.running
        return {"taskArns": [t["taskArn"] for t in source]}

    def describe_tasks(self, cluster, tasks):
        return {"tasks": [t for t in self.stopped + self.running if t["taskArn"] in tasks]}

    def describe_task_definition(self, taskDefinition):
        return {"taskDefinition": {"containerDefinitions": [{"name": "mlflow", "logConfiguration": {"options": {
            "awslogs-group": "/ecs/sepsis-dev-mlflow", "awslogs-stream-prefix": "mlflow"}}}]}}


class FakeLogs:
    def get_log_events(self, logGroupName, logStreamName, limit, startFromHead):
        self.asked = (logGroupName, logStreamName)
        return {"events": [{"message": "ERROR mlflow.cli: No module named 'psycopg'\n"}]}


class FakeCloudFormation:
    def __init__(self):
        self.cancelled = []

    def cancel_update_stack(self, StackName):
        self.cancelled.append(StackName)


def service(name="sepsis-dev-mlflow", running=0, desired=1):
    return {"serviceName": name, "status": "ACTIVE", "runningCount": running, "desiredCount": desired, "pendingCount": 0,
            "events": [{"message": f"(service {name}) has started 1 tasks"}]}


def stopped_task(n, when=NOW):
    return {"taskArn": f"arn:aws:ecs:r:1:task/c/t{n}", "taskDefinitionArn": "td", "stoppedAt": when,
            "stoppedReason": "Essential container in task exited", "containers": [{"name": "mlflow", "exitCode": 1}]}


def running_task(n, health="HEALTHY"):
    return {"taskArn": f"arn:aws:ecs:r:1:task/c/r{n}", "lastStatus": "RUNNING", "healthStatus": health}


class WatchTests(unittest.TestCase):
    def run_watch(self, ecs, **kwargs):
        out, cloudformation = [], FakeCloudFormation()
        code = watcher.watch(ecs, FakeLogs(), cloudformation, "sepsis-dev", out=out.append, sleep=lambda _: None,
                             clock=lambda: NOW - timedelta(minutes=5), **kwargs)
        return code, out, cloudformation

    def test_a_stopped_task_is_explained_with_its_log(self):
        code, out, cf = self.run_watch(FakeEcs([service()], stopped=[stopped_task(1)]), once=True)
        text = "\n".join(out)
        self.assertEqual(code, 0)
        self.assertIn("Essential container in task exited", text)
        self.assertIn("exit code 1", text)
        self.assertIn("No module named 'psycopg'", text)
        self.assertEqual(cf.cancelled, [])

    def test_a_crash_loop_cancels_the_update_and_exits_2(self):
        code, out, cf = self.run_watch(FakeEcs([service()], stopped=[stopped_task(n) for n in range(3)]))
        self.assertEqual(code, 2)
        self.assertEqual(cf.cancelled, ["sepsis-dev"])
        self.assertIn("crash-looping", "\n".join(out))

    def test_no_cancel_flag_only_reports(self):
        code, _, cf = self.run_watch(FakeEcs([service()], stopped=[stopped_task(n) for n in range(3)]), cancel=False)
        self.assertEqual(code, 2)
        self.assertEqual(cf.cancelled, [])

    def test_tasks_from_before_this_deploy_do_not_count(self):
        old = [stopped_task(n, NOW - timedelta(hours=1)) for n in range(5)]
        code, out, cf = self.run_watch(FakeEcs([service()], stopped=old), once=True)
        self.assertEqual(code, 0)
        self.assertEqual(cf.cancelled, [])
        self.assertNotIn("stopped:", "\n".join(out))

    def test_a_running_service_is_never_a_crash_loop(self):
        code, _, cf = self.run_watch(FakeEcs([service(running=1)], stopped=[stopped_task(n) for n in range(5)]), once=True)
        self.assertEqual(code, 0)
        self.assertEqual(cf.cancelled, [])

    def test_a_service_that_does_not_exist_yet_is_fine(self):
        code, out, _ = self.run_watch(FakeEcs([]), once=True)
        self.assertEqual((code, out), (0, []))


class ReadyTests(unittest.TestCase):
    def run_ready(self, ecs, ticks=2):
        out, clock = [], iter(range(0, 10000, 10))
        code = watcher.ready(ecs, FakeLogs(), "sepsis-dev", timeout=ticks * 10, interval=0, sleep=lambda _: None,
                             clock=lambda: next(clock), out=out.append)
        return code, "\n".join(out)

    def test_running_and_healthy_passes(self):
        code, text = self.run_ready(FakeEcs([service(running=1)], running=[running_task(1)]))
        self.assertEqual(code, 0)
        self.assertIn("running 1/1 and healthy", text)

    def test_a_switched_off_service_is_skipped(self):
        code, text = self.run_ready(FakeEcs([service(running=0, desired=0)]))
        self.assertEqual(code, 0)
        self.assertIn("switched off", text)

    def test_an_unhealthy_task_never_becomes_ready_and_is_diagnosed(self):
        code, text = self.run_ready(FakeEcs([service(running=1)], running=[running_task(1, "UNHEALTHY")], stopped=[stopped_task(1, NOW + timedelta(hours=9))]))
        self.assertEqual(code, 1)
        self.assertIn("NOT READY", text)

    def test_a_service_that_never_starts_times_out_with_the_stopped_reason(self):
        code, text = self.run_ready(FakeEcs([service(running=0)], stopped=[stopped_task(1, NOW + timedelta(hours=9))]))
        self.assertEqual(code, 1)
        self.assertIn("No module named 'psycopg'", text)


if __name__ == "__main__":
    unittest.main()
