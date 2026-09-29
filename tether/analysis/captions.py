"""Caption registry: every figure naming a result states that result's hedge verbatim."""

STATUS = {
    "Prop. 1": "PROV under (H1),(H2)",
    "Prop. 2": "PROV",
    "Prop. 3": "PROV given (H3)-(H5)",
    "Prop. 4": "PROV given (H4)",
    "Cor. 5": "CONJ (leading order)",
    "Thm 6": "CONJ",
    "Thm 7": "CONJ",
    "Cor. 7.1": "CONJ",
    "Cor. 7.2": "PROV given Delta_max",
    "Prop. 8": "PROV under (H1),(H2)",
    "Prop. 9": "CONJ",
    "Prop. 10": "PROV by construction",
    "Prop. 11": "APPROX, domain measured",
    "Prop. 12": "PROV to leading order",
}


def hedge(result: str) -> str:
    return f"{result} [{STATUS[result]}]"


CAPTIONS: dict[str, str] = {}


def register(figure_id: str, text: str) -> str:
    CAPTIONS[figure_id] = text
    return text
