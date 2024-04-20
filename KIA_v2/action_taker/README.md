# Automated Meeting Summarization and Task Generation

## Overview
This script uses the `crewai` library to automate the summarization of meeting transcripts, generate actionable tasks from the summaries, and create JIRA tasks for these actions using CURL scripts.

## Setup
The script initializes the environment for the OpenAI API with the specified base URL, model, and an API key.
You can run the model locally by tweaking the agent to include a locally intalled llm.
In this set up we leverage the groq api. We do it by providing the groq api key we can get from our own profile once we sing up on groq (https://console.groq.com/keys).

### Dependencies
To run this script, you will need to install the following Python libraries:
- `crewai`: This library is essential for defining and managing the agents and tasks.
- `os`: Standard library, typically included with Python, used here to set environment variables.

You can install the `crewai` library using pip if it's available:
```bash
pip install crewai
```

## Considerations
Groq in combination with the model I used (LLama3-80b) supports only a maximum of 5000 tokens. Therefore, for long files, I recommend to have an AI reduce the size of the text file before hand.

## Components

### Agents
- **Summarizer:** Summarizes the meeting transcript into bullet points with detailed descriptions.
- **Actioner:** Creates action items based on the summarized bullet points.
- **PM:** Generates CURL scripts for creating JIRA tasks based on the action items.

### Tasks
Each agent is assigned a specific task:
- The **Summarizer** processes the meeting transcript into a bullet-point summary.
- The **Actioner** reviews the summaries to create actionable tasks.
- The **PM** takes these actions and creates JIRA tasks.

## Process
The tasks are handled sequentially by a crew of agents, ensuring each task's output is used as input for the next.

## Execution
The script culminates in executing these tasks, with the final output being the CURL scripts for JIRA task creation, which are then printed.

## Conclusion
This automated process enhances efficiency by reducing manual intervention in meeting management and task tracking.

## Next steps
Add an agent that runs the curl(s).
