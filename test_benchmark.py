import unittest
import os
import sys
import subprocess

class TestWhisperBenchmark(unittest.TestCase):
    def setUp(self):
        # Ensure the test audio file exists
        self.test_audio_file = "test_audio/test_voice.mp3"
        self.assertTrue(
            os.path.exists(self.test_audio_file),
            f"Test audio file {self.test_audio_file} not found"
        )

        # Ensure the benchmark script exists
        self.benchmark_script = "whisper_benchmark.py"
        self.assertTrue(
            os.path.exists(self.benchmark_script),
            f"Benchmark script {self.benchmark_script} not found"
        )

    def test_benchmark_script_runs(self):
        """Test that the benchmark script runs without errors with the default model."""
        result = subprocess.run(
            [
                sys.executable, self.benchmark_script,
                "--audio", self.test_audio_file,
                "--models", "mlx-community/parakeet-tdt-0.6b-v3",
            ],
            capture_output=True,
            text=True,
            timeout=300  # Allow up to 5 minutes for model download + transcription
        )
        self.assertEqual(result.returncode, 0, f"Benchmark script failed with error: {result.stderr}")

        # Check for the table output which indicates successful completion
        self.assertIn("Model", result.stdout, "Model table header not found in output")
        self.assertIn("Real-time Factor", result.stdout, "Real-time factor not reported")
        self.assertIn("Fastest model:", result.stdout, "Fastest model summary not found")

    def test_benchmark_alias(self):
        """Test that the benchmark script accepts model aliases."""
        result = subprocess.run(
            [
                sys.executable, self.benchmark_script,
                "--audio", self.test_audio_file,
                "--models", "parakeet-v3",
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        self.assertEqual(result.returncode, 0, f"Benchmark script failed with error: {result.stderr}")

    def test_simple_mode(self):
        """Test that the --simple flag works."""
        result = subprocess.run(
            [
                sys.executable, self.benchmark_script,
                "--audio", self.test_audio_file,
                "--simple",
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        self.assertEqual(result.returncode, 0, f"Simple mode failed with error: {result.stderr}")
        self.assertIn("Transcription result:", result.stdout)


if __name__ == "__main__":
    unittest.main()
