class SubProcessVectorEnvException(Exception):
    """SubProcessVectorEnvException."""

    def __init__(self, message: str):
        """__init__.

        Args:
            message (str): message
        """
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """__str__."""
        return f"[SubProcVecEnv Exception]: {self.message}"
