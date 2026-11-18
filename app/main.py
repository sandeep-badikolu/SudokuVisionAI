"""Process entry point.

This module only identifies where the application will start. Image loading,
digit recognition, validation, and solving are not called from here yet.
"""


def main() -> None:
    """Point to the Streamlit application."""
    print("Launch the UI with: python -m streamlit run app/ui/streamlit_app.py")


if __name__ == "__main__":
    main()
