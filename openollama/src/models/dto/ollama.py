from .base import Base

from dataclasses import dataclass, field

@dataclass
class OllamaOptions(Base):
    mirostat: int = 0
    mirostat_eta: float = 0.1
    mirostat_tau: float = 5.0
    num_ctx: int = 2048
    num_thread: int = 4
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    temperature: float = 0.8
    tfs_z: int = 1
    num_predict: int = 128
    top_k: int = 40
    top_p: float = 0.9


@dataclass
class OllamaParams(Base):
    model: str
    prompt: str
    options: OllamaOptions
    system: str = ""
    template: str = ""
    context: list[int] = field(default_factory=list)

    def generate_post_data(self):
        d = {k: v for k, v in self.__dict__.copy().items() if v}
        d.update({"options": self.options.__dict__.copy()})
        return d

    @classmethod
    def parse(_, data: dict) -> "OllamaParams":
        return super().parse(data, options=OllamaOptions.parse)
