#!/bin/bash
# download grammar models for parsing dependencies

# cmcc, guardian, imdb, reuters
poetry run python -c "import stanza; stanza.download('en')"

# pan15
poetry run python -c "import stanza; stanza.download('es')"
poetry run python -c "import stanza; stanza.download('nl')"
poetry run python -c "import stanza; stanza.download('it')"

# pan18
poetry run python -c "import stanza; stanza.download('fr')"
poetry run python -c "import stanza; stanza.download('pl')"
