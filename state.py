from typing import TypedDict, List, Annotated
import operator

class RevisaoState(TypedDict):
    tema: str
    tipo_revisao: str
    chunks_relevantes: List[str]
    snippets_tecnicos: List[dict]
    urls_tecnicos: List[str]
    plano_atual: str
    historico_entrevista: Annotated[List[tuple], operator.add]
    perguntas_feitas: int
    max_perguntas: int
    plano_final: str
    plano_final_path: str
    status: str


class RevisaoState(TypedDict):
    tema: str
    tipo_revisao: str
    chunks_relevantes: List[str]
    snippets_tecnicos: List[dict]
    urls_tecnicos: List[str]
    plano_atual: str
    historico_entrevista: Annotated[List[tuple], operator.add]
    perguntas_feitas: int
    max_perguntas: int
    plano_final: str
    plano_final_path: str
    status: str

class EscritaTecnicaState(TypedDict):
    tema: str
    resumo_plano: str
    secoes: List[dict]
    caminho_plano: str
    secoes_escritas: List[dict]
    refs_urls: List[str]
    refs_imagens: List[dict]
    resumo_acumulado: str
    react_log: List[str]
    stats_verificacao: List[dict]
    status: str