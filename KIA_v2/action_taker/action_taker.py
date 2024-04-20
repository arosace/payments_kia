from crewai import Agent, Task, Crew, Process
import os

os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] = "llama3-70b-8192"
os.environ["OPENAI_API_KEY"] = "YOUR_GROQ_API_KEY"

file_path = './data/<file.whatever>'

# Open the file and read its contents
with open(file_path, 'r', encoding='utf-8') as file:
    meeting_recording = file.read()

is_verbose = True

summarizer = Agent(
    role="conversation summarizer",
    goal="accurately extrapolate from the meeting transcription the main topics that have been raised in the meeting. Write a bulletpoint list of the topics. Add a description to each of them.",
    backstory="You are an AI assistant whose only job is to read a meeting transcript and summarize its content in a bulletpoint list based on the topics raised in the meeting. Your job is to help the development teams have an overview of what was talked aboutduring the meeting.",
    verbose=is_verbose,
    allow_delegation=False)

actioner = Agent(
    role="action taker",
    goal="Based on the bullet point list of topics provided and on their description. Extrapolate at least on action item from each of the bullet point. If the bullet point description is too vague write 'gather more info' as action item.",
    backstory="You are an AI assistant whose job is to read a list of topics and their description and come up with at least an action item for each of those. The bullet point list will be provided by the 'summarizer' agent.",
    verbose=is_verbose,
    allow_delegation=False)

pm = Agent(
    role="JIRA task creator",
    goal="Based on the list of action items. Create a curl script with all the information necessary to create a JIRA ticket using the JIRA api.",
    backstory="You are an AI assistant whose job is to create a JIRA task via curl script for each action item provided by the 'actioner' agent.",
    verbose=is_verbose,
    allow_delegation=False)

summarize_meeting = Task(
    description=f"Summarize the following meeting transcript: '{meeting_recording}' into a bullet point list. Each bullet point needs to have an exaustive description of the topic based on the meeting transcript",
    agent = summarizer,
    expected_output="A bullet point list"
)

write_action_items = Task(
    description=f"Generate action items for the meeting transcript '{meeting_recording}' based on the bullet point list provided by the 'summarizer' agent.",
    agent = actioner,
    expected_output="A list of action items based on the bullet point list provided by the 'summarizer' agent. At least on action item per bullet point."
)

task_creator_JIRA = Task(
    description=f"Generate curl scripts for each of the action items for the meeting transcript '{meeting_recording}' provided by the 'actioner' agent. ",
    agent = pm,
    expected_output="A list curl scripts to create tasks in jira via the jira api. One curls script per action item provided by the 'actioner' agent."
)

crew = Crew(
    agents = [summarizer, actioner,pm],
    tasks = [summarize_meeting, write_action_items, task_creator_JIRA],
    verbose = 2,
    process = Process.sequential
)

output = crew.kickoff()
print(output)