class UnexpectedMode(ValueError):
    def __init__(self, mode: str) -> None:
        super().__init__(
            f"Unexpected mode - found '{mode}' but must be 'image' or 'mesh'"
        )
