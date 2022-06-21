from click.testing import CliRunner

import minidalle2.trainer.commands.train_clip_command as train_clip_command


class TestTrainCommand:
    def test_train_command_no_steps(self):
        runner = CliRunner()
        result = runner.invoke(
            train_clip_command.execute,
            [
                "--n-epochs",
                "1",
            ],
        )
        assert result.exit_code == 0
        assert result.output == "Done.\n"

    def test_train_command_default(self):
        runner = CliRunner()
        result = runner.invoke(
            train_clip_command.execute,
        )
        assert result.exit_code == 0
        assert result.output == "Done.\n"
