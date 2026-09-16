from pathlib import Path

from prototype.presentation import install_player_pulse_theme


def test_sidebar_navigation_options_fill_the_sidebar_width(monkeypatch):
    rendered = {}

    def capture_markdown(body, **_kwargs):
        rendered["body"] = body

    monkeypatch.setattr("prototype.presentation.st.markdown", capture_markdown)

    install_player_pulse_theme(Path("prototype/assets/pixel-world-background.png"))

    assert """[data-testid="stSidebar"] div[role="radiogroup"] {
            align-items: stretch;
            gap: 0;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            box-sizing: border-box;
            width: 100%;""" in rendered["body"]
