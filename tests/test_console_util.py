from rich.console import Console

from utils.console_util import CaptionAndRateLayout


def _render_rating_chart_text(ratings):
    layout = object.__new__(CaptionAndRateLayout)
    chart = layout.create_rating_chart(ratings)
    console = Console(force_terminal=False, width=120, record=True)
    console.print(chart)
    return console.export_text()


def test_rating_chart_shortens_long_dimension_labels():
    text = _render_rating_chart_text(
        {
            "Character Portrayal & Posing": 8,
            "Overall Impact & Uniqueness": 9,
        }
    )

    assert "Character:" in text
    assert "Overall:" in text
    assert "Character Portrayal" not in text
    assert "Overall Impact" not in text


def test_rating_chart_shortens_grok_snake_case_dimension_labels():
    text = _render_rating_chart_text(
        {
            "costume_presentation_accuracy": 8,
            "character_portrayal_posing": 5,
            "setting_environment_integration": 5,
            "lighting_mood": 4,
            "composition_framing": 6,
            "storytelling_concept": 3,
            "level_of_sexy": 1,
            "figure_silhouette_fit": 7,
            "overall_impact_uniqueness": 6,
        }
    )

    assert "Character:" in text
    assert "Overall:" in text
    assert "character_portrayal_posing" not in text
    assert "overall_impact_uniqueness" not in text
