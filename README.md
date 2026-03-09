# Autonomous Robot Navigation via VLA Multi-Agent Orchestration

This project implements a cutting-edge **Vision-Language-Action (VLA)** pipeline for autonomous robotic navigation. It leverages **Multi-Agent AI** to transform raw visual environmental data into safe, machine-executable navigation constraints and waypoints.

## 🚀 Overview

The system bridges the gap between high-level semantic scene understanding and low-level robotic control. By utilizing state-of-the-art Large Multimodal Models (LMMs) and specialized AI Agents, the project automates the transition from "seeing" a room to "safely navigating" it in theory.

### Key Features
* **Visual Grounding:** Uses the **Qwen2.5-VL-7B** model to perform semantic analysis of indoor environments.
* **Agentic Reasoning:** Implements a **CrewAI** multi-agent architecture to decompose complex navigation tasks.
* **Safety Auditing:** A dedicated agent performs safety verification against detected physical obstacles (rugs, furniture, etc.).
* **Formal Translation:** Converts high-level reasoning into structured **JSON constraints** ready for robotic middleware (e.g., ROS2, MQTT).

## 🧠 Architecture

The workflow follows a sequential multi-agent process:

1.  **Perception Specialist:** Analyzes the grounded visual facts to identify navigation constraints (e.g., surface traction, obstacle proximity).
2.  **Safety Auditor:** Generates trajectory waypoints `[x, y]` while justifying safety protocols based on the perception data.
3.  **Formal Verification Unit:** Translates the natural language safety rules and coordinates into machine-readable JSON formats.

## 🛠️ Tech Stack

* **Logic Engine:** [Llama-4-Scout-17B](https://groq.com/) via Groq API.
* **Vision Engine:** [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) via HuggingFace Inference.
* **Agent Framework:** [CrewAI](https://www.crewai.com/) for task orchestration.
* **Language:** Python 3.10+

## 📋 Prerequisites

- Python 3.10 or higher
- API Keys for **Groq** and **HuggingFace**
