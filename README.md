# AIY Voice Assistant

This project uses the Google AIY Voice Kit to create a voice assistant that interacts with the OpenAI GPT-4o model.

## Setup

### Prerequisites

- Raspberry Pi with Raspbian installed
- Google AIY Voice Kit assembled
- Follow oficial instructions to flash microsd card with AIY Voice Kit image

### Installation

1. Clone the repository and navigate to the project directory.

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv --system-site-packages venv
    source venv/bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory with your OpenAI API key:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key_here
    ```

### Running the Project

Run the main script:

```bash
python main.py
