"""Check what the scripts and pipelines send to AWS against AWS's own API models (the service definitions in botocore).

Calls made with a name or structure the real API does not have, and response fields the code reads that the real API
never returns, fail here instead of in an AWS account. This does not replace the documentation checks recorded in
docs/aws-assumptions.md (behaviour, quotas, defaults); it covers names and shapes.
"""

import sys
import unittest
from pathlib import Path

import yaml

try:
    import botocore.session
    from botocore import xform_name
    from botocore.validate import ParamValidator
except ImportError:  # the AWS API models ship with botocore (aws extra)
    botocore = None

NEEDS_BOTOCORE = unittest.skipUnless(botocore, "needs `pip install -e .[aws]`")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aws_approval as approval  # noqa: E402
import aws_power as power  # noqa: E402
import aws_run_task as run_task  # noqa: E402

SESSION = botocore.session.get_session() if botocore else None


def model(service: str):
    return SESSION.get_service_model(service)


def operations(service: str) -> dict:
    return {xform_name(name): name for name in model(service).operation_names}


def check_response(value, shape, where: str, problems: list) -> None:
    """Every key of a (fake) response must be a member of the real output shape."""
    if shape is None:
        return
    if shape.type_name == "structure" and isinstance(value, dict):
        for key, item in value.items():
            if key not in shape.members:
                problems.append(f"{where}: the real response has no field {key!r}")
            else:
                check_response(item, shape.members[key], f"{where}.{key}", problems)
    elif shape.type_name == "list" and isinstance(value, list):
        for index, item in enumerate(value):
            check_response(item, shape.member, f"{where}[{index}]", problems)


class Checked:
    """A fake client that refuses calls the real API would refuse, and responses it would never send."""

    def __init__(self, fake, service: str, problems: list):
        self._fake, self._service, self._problems = fake, service, problems
        self._ops, self._model = operations(service), model(service)

    def __getattr__(self, name):
        target = getattr(self._fake, name)
        if name not in self._ops:
            return target
        operation = self._model.operation_model(self._ops[name])

        def call(*args, **kwargs):
            report = ParamValidator().validate(kwargs, operation.input_shape) if operation.input_shape else None
            if report is not None and report.has_errors():
                self._problems.append(f"{self._service}.{name}: {report.generate_report()}")
            response = target(*args, **kwargs)
            if isinstance(response, dict):
                check_response(response, operation.output_shape, f"{self._service}.{name} response", self._problems)
            return response

        return call


class CheckedSession:
    def __init__(self, session, services: dict, problems: list):
        self._clients = {name: Checked(session.client(name), services.get(name, name), problems) for name in services}

    def client(self, name):
        return self._clients[name]


@NEEDS_BOTOCORE
class TheCheckBitesTests(unittest.TestCase):
    def test_a_wrong_name_or_an_invented_response_field_is_reported(self):
        problems = []

        class Ecs:
            def update_service(self, **kwargs):
                return {"service": {"desiredCountt": 1}}

        Checked(Ecs(), "ecs", problems).update_service(cluster="c", service="s", desiredCounts=0)
        self.assertEqual(len(problems), 2)
        self.assertIn("desiredCounts", problems[0])
        self.assertIn("desiredCountt", problems[1])
        wrong = []
        check_asl({"TaskDefinitionArn": "x"}, model("ecs").operation_model("RunTask").input_shape, "state", wrong)
        self.assertIn("TaskDefinitionArn", wrong[0])


@NEEDS_BOTOCORE
class ClientCallTests(unittest.TestCase):
    def test_the_power_script_only_makes_calls_the_real_apis_accept(self):
        import test_power

        problems = []
        world = test_power.World(db_status="stopped", running=0)
        session = CheckedSession(world.session(), {"ecs": "ecs", "rds": "rds", "ec2": "ec2", "cloudformation": "cloudformation"}, problems)

        def fake_sleep(_):
            if world.db_status == "starting":
                world.db_status = "available"
        tool = power.Power(session, "sepsis-dev", sleep=fake_sleep, timeout=100000)
        tool.start(allow_cidr="203.0.113.7/32")
        power.Power(session, "sepsis-dev", sleep=fake_sleep, timeout=100000).stop()
        power.Power(session, "sepsis-dev").status()
        self.assertEqual(problems, [])

    def test_the_task_runner_only_makes_calls_the_real_apis_accept(self):
        import test_approval

        problems = []
        ecs = test_approval.FakeEcs()
        fake = test_approval.FakeSession(ecs)
        session = CheckedSession(fake, {"ecs": "ecs", "cloudformation": "cloudformation", "logs": "logs"}, problems)
        run_task.run(session, "sepsis-dev", role="approver", image_tag="training-abc", env=["HP_mode=promote"],
                     family_suffix="approver", region="ap-southeast-2", sleep=lambda _: None, out=lambda *_: None)
        self.assertEqual(problems, [])

    def test_the_approval_script_only_makes_calls_the_real_apis_accept(self):
        import test_approval

        problems = []
        sfn = Checked(test_approval.FakeStepFunctions(), "stepfunctions", problems)
        approval.decide(sfn, "arn:aws:states:ap-southeast-2:1:stateMachine:m", "approve", "alice")
        approval.decide(Checked(test_approval.FakeStepFunctions(), "stepfunctions", problems),
                        "arn:aws:states:ap-southeast-2:1:stateMachine:m", "reject", "bob", reason="no")
        self.assertEqual(problems, [])


def pascal(name: str) -> str:
    return name[:1].upper() + name[1:]


def check_asl(params, shape, where: str, problems: list) -> None:
    """Step Functions writes every parameter in PascalCase, whatever the API's own spelling (ECS is camelCase,
    SageMaker and SNS are already PascalCase): each name must match a member of the real input shape."""
    if shape is None:
        return
    if shape.type_name == "structure" and isinstance(params, dict):
        members = {pascal(member): member for member in shape.members}
        for key, value in params.items():
            name = key[:-2] if key.endswith(".$") else key
            if name not in members:
                problems.append(f"{where}: the real API has no parameter {name!r} (it has {sorted(members)[:8]}...)")
            else:
                check_asl(value, shape.members[members[name]], f"{where}.{name}", problems)
    elif shape.type_name == "list" and isinstance(params, list):
        for index, item in enumerate(params):
            check_asl(item, shape.member, f"{where}[{index}]", problems)


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_multi_constructor("!", lambda loader, suffix, node: loader.construct_scalar(node) if isinstance(node, yaml.ScalarNode)
                              else (loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else loader.construct_mapping(node)))


@NEEDS_BOTOCORE
class StateMachineParameterTests(unittest.TestCase):
    def states(self, name: str) -> dict:
        template = yaml.load((ROOT / "infra" / "cloudformation" / name).read_text(encoding="utf-8"), Loader=_Loader)
        return template["Resources"]["TrainingPipeline"]["Properties"]["Definition"]["States"]

    def test_every_task_state_uses_parameter_names_that_exist_in_the_real_api(self):
        ecs, sagemaker = model("ecs"), model("sagemaker")
        targets = {
            "arn:aws:states:::ecs:runTask.sync": ecs.operation_model("RunTask").input_shape,
            "arn:aws:states:::aws-sdk:ecs:registerTaskDefinition": ecs.operation_model("RegisterTaskDefinition").input_shape,
            "arn:aws:states:::sagemaker:createTrainingJob.sync": sagemaker.operation_model("CreateTrainingJob").input_shape,
            "arn:aws:states:::sns:publish": model("sns").operation_model("Publish").input_shape,
            "arn:aws:states:::sns:publish.waitForTaskToken": model("sns").operation_model("Publish").input_shape,
        }
        checked = 0
        for name in ("pipeline.yaml", "pipeline-ecs.yaml"):
            problems = []
            for state, body in self.states(name).items():
                if body.get("Type") != "Task":
                    continue
                shape = targets.get(body["Resource"])
                self.assertIsNotNone(shape, f"{name} {state}: a resource this test does not know: {body['Resource']}")
                check_asl(body.get("Parameters", {}), shape, f"{name} {state}", problems)
                checked += 1
            self.assertEqual(problems, [])
        self.assertGreater(checked, 10)

    def test_the_sns_message_attribute_has_the_shape_the_real_api_defines(self):
        attributes = model("sns").operation_model("Publish").input_shape.members["MessageAttributes"]
        self.assertEqual(attributes.value.members["DataType"].type_name, "string")
        self.assertIn("StringValue", attributes.value.members)

    def test_the_fargate_task_sizes_in_the_templates_are_valid_ecs_numbers(self):
        # The task-size table itself is checked against the documentation in docs/aws-assumptions.md.
        for state in self.states("pipeline-ecs.yaml").values():
            if state.get("Resource") == "arn:aws:states:::aws-sdk:ecs:registerTaskDefinition":
                self.assertIn(state["Parameters"]["Cpu"], ("${Cpu}",))
                self.assertIn(state["Parameters"]["Memory"], ("${Memory}",))


if __name__ == "__main__":
    unittest.main()
