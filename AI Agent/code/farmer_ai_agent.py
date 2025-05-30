import os
import json
import requests
from typing import Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pickle
import gradio as gr

# Clear any existing environment variable to isolate the .env file
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# Load environment variable
load_dotenv()

# OPENAI API key from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


# --------------------------------------------------------------
# Define response formats using Pydantic models
# --------------------------------------------------------------
class Weather(BaseModel):
    temperature: float = Field(description="Current temperature in Celsius")
    wind_speed: float = Field(description="Current wind speed in m/s")


class SoilData(BaseModel):
    moisture_level: float = Field(description="Soil moisture percentage (0-100)")
    temperature: float = Field(description="Soil temperature in Celsius")


class IrrigationResponse(BaseModel):
    temperature: float = Field(description="Current air temperature in Celsius")
    soil_moisture: float = Field(description="Current soil moisture percentage")
    irrigation_needed: bool = Field(description="Whether irrigation is recommended")
    water_amount: float = Field(
        description="Recommended water amount in liters per square meter"
    )
    responses: str = Field(
        description="Natural language response with irrigation advice"
    )


# --------------------------------------------------------------
# Define tools (functions)
# --------------------------------------------------------------
def get_weather(latitude: float, longitude: float) -> dict:
    """Fetch current weather data for given coordinates."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m"
    )
    data = response.json()
    return {
        "temperature": data["current"]["temperature_2m"],
        "wind_speed": data["current"]["wind_speed_10m"],
    }


def get_soil_data(sensor_id: str) -> dict:
    """Simulated soil sensor data (replace with real sensor API/integration)"""
    import random

    return {
        "moisture_level": random.uniform(20, 80),
        "temperature": random.uniform(15, 30),
    }


def control_irrigation(water_amount: float) -> dict:
    """Simulated irrigation control"""
    return {"status": "success", "water_amount": water_amount}


def call_function(name: str, args: dict) -> dict:
    if name == "get_weather":
        return get_weather(**args)
    elif name == "get_soil_data":
        return get_soil_data(**args)
    elif name == "control_irrigation":
        return control_irrigation(**args)
    return {}


# --------------------------------------------------------------
# Define tools for the AI agent
# --------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the location"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the location"
                    },
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_soil_data",
            "description": "Get current soil sensor readings",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "sensor_id": {
                        "type": "string",
                        "description": "Sensor ID for the soil device"
                    },
                },
                "required": ["sensor_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_irrigation",
            "description": "Activate irrigation system with specified water amount",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "water_amount": {
                        "type": "number",
                        "description": "Water amount to apply in liters per square meter"
                    },
                },
                "required": ["water_amount"],
                "additionalProperties": False,
            },
        },
    },
]


# --------------------------------------------------------------
# Learning mechanism
# --------------------------------------------------------------
class LearningAgent:
    def __init__(self):
        self.data_file = "irrigation_learning_data.pkl"
        self.default_rules = {
            "cotton": {
                "min_moisture": 50,
                "base_water": 6.0,
                "temp_threshold": 28,
                "wind_adjust": 0.8,
            },
            "corn": {
                "min_moisture": 40,
                "base_water": 5.0,
                "temp_threshold": 25,
                "wind_adjust": 0.9,
            },
            "soybeans": {
                "min_moisture": 45,
                "base_water": 5.5,
                "temp_threshold": 27,
                "wind_adjust": 0.85,
            },
            "peanut": {
                "min_moisture": 45,
                "base_water": 5.5,
                "temp_threshold": 26,
                "wind_adjust": 0.85,
            },
        }
        self.default_conditions = {
            "last_moisture": 50.0,
            "last_temp": 25.0,
            "last_water": 5.0,
        }
        self.learned_rules = self.load_learned_data()
        self.last_conditions = self.load_last_conditions()
        print("Loaded learned rules:", self.learned_rules)
        print("Loaded last conditions:", self.last_conditions)

    def load_learned_data(self) -> Dict:
        """Load previously saved rules or return defaults if file doesn't exist."""
        try:
            with open(self.data_file, "rb") as f:
                data = pickle.load(f)
                # Ensure only valid crop keys are loaded
                return {
                    crop: data.get(crop, self.default_rules[crop])
                    for crop in self.default_rules
                }
        except (FileNotFoundError, pickle.PickleError):
            return self.default_rules.copy()

    def load_last_conditions(self) -> Dict:
        """Load last conditions or return defaults."""
        try:
            with open(self.data_file, "rb") as f:
                data = pickle.load(f)
                return {
                    "last_moisture": data.get("last_moisture", 50.0),
                    "last_temp": data.get("last_temp", 25.0),
                    "last_water": data.get("last_water", 5.0),
                }
        except (FileNotFoundError, pickle.PickleError):
            return self.default_conditions.copy()

    def save_learned_data(self):
        """Save current rules and last conditions to file."""
        data = self.learned_rules.copy()
        data.update(self.last_conditions)
        with open(self.data_file, "wb") as f:
            pickle.dump(data, f)

    def update_rules(
        self,
        crop: str,
        moisture: float,
        temp: float,
        water_used: float,
        feedback: float,
    ):
        """Update rules based on feedback."""
        rules = self.learned_rules[crop]
        if feedback < 0.7:  # Adjust if outcome was suboptimal
            if moisture < rules["min_moisture"]:  # Too dry
                rules["min_moisture"] -= 1.0
                rules["base_water"] += 0.2
            elif moisture > rules["min_moisture"] + 10:  # Too wet
                rules["min_moisture"] += 1.0
                rules["base_water"] -= 0.2
            if temp > rules["temp_threshold"] and feedback < 0.6:  # Too hot
                rules["temp_threshold"] -= 0.5
        self.save_learned_data()


# --------------------------------------------------------------
# System prompt
# --------------------------------------------------------------
system_prompt = """
You are an adaptive irrigation AI agent. To make irrigation decisions:
1. Always call the `get_weather` and `get_soil_data` tools to gather current conditions for the specified location and sensor.
2. Assess conditions for the specified crop (cotton, corn, soybeans, peanut) using the provided rules.
3. Adapt irrigation decisions based on learned rules:
   - Cotton: 50% moisture, 6L/m² base
   - Corn: 40% moisture, 5L/m² base
   - Soybeans: 45% moisture, 5.5L/m² base
   - Peanut: 45% moisture, 5.5L/m² base
   - Adjust water: +15-20% if temperature exceeds crop threshold, -10-20% if wind > 5 m/s.
4. Learn from feedback to refine thresholds and water amounts.
5. Return a detailed, crop-specific recommendation in the IrrigationResponse format.
6. If irrigation is recommended, call the `control_irrigation` tool with the recommended water amount.
"""


# --------------------------------------------------------------
# Core irrigation logic
# --------------------------------------------------------------
def manage_irrigation(
    location: str,
    crop: str,
    agent: LearningAgent,
    sensor_id: str = "sensor_001",
    latitude: float = 31.4505,  # Default: Tifton, GA
    longitude: float = -83.5085,
    feedback: Optional[float] = None,
):
    # Debug: Print the crop input
    print(f"Received crop input: '{crop}'")

    # Normalize crop to lowercase for consistency
    crop = crop.lower().strip()

    # Check if crop is supported
    if crop not in agent.learned_rules:
        supported_crops = ", ".join(
            key for key in agent.learned_rules.keys() if key in agent.default_rules
        )
        return IrrigationResponse(
            temperature=0,
            soil_moisture=0,
            irrigation_needed=False,
            water_amount=0,
            responses=f"Error: {crop} not recognized. Supported crops: {supported_crops}",
        )

    # Update rules if feedback is provided
    if feedback is not None:
        agent.update_rules(
            crop,
            agent.last_conditions.get("last_moisture", 50),
            agent.last_conditions.get("last_temp", 25),
            agent.last_conditions.get("last_water", 5),
            feedback,
        )

    # Initialize messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Should I irrigate my {crop} field at {location} today? Current rules: {agent.learned_rules[crop]}",
        },
    ]

    # Initial call to check for tool calls
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # Convert tool_calls to JSON-serializable format
    tool_calls = None
    if completion.choices[0].message.tool_calls:
        tool_calls = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            for tool_call in completion.choices[0].message.tool_calls
        ]

    # Append the assistant's response
    assistant_message = {
        "role": "assistant",
        "content": completion.choices[0].message.content,
        "tool_calls": tool_calls
    }
    messages.append(assistant_message)

    # Process tool calls if they exist
    if tool_calls:
        for tool_call in completion.choices[0].message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            if name == "get_weather":
                args = {"latitude": latitude, "longitude": longitude}
            elif name == "get_soil_data":
                args = {"sensor_id": sensor_id}
            result = call_function(name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    # Debug: Print messages before final API call
    print("Messages sent to final API call:", json.dumps(messages, indent=2))

    # Final response with decision
    completion_final = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        response_format=IrrigationResponse,
    )
    response = completion_final.choices[0].message.parsed

    # Store latest conditions
    agent.last_conditions["last_moisture"] = response.soil_moisture
    agent.last_conditions["last_temp"] = response.temperature
    agent.last_conditions["last_water"] = response.water_amount
    agent.save_learned_data()

    return response


# --------------------------------------------------------------
# Gradio Interface
# --------------------------------------------------------------
agent = LearningAgent()


def check_irrigation(crop, location, latitude, longitude):
    # Normalize crop input
    crop = crop.lower().strip()
    response = manage_irrigation(
        location, crop, agent, latitude=latitude, longitude=longitude
    )
    output = (
        f"Temperature: {response.temperature}°C\n"
        f"Soil Moisture: {response.soil_moisture:.2f}%\n"
        f"Irrigation Needed: {'Yes' if response.irrigation_needed else 'No'}\n"
        f"Water Amount: {response.water_amount} L/m²\n"
        f"Recommendation: {response.responses}"
    )
    return output


def provide_feedback(crop, feedback_score):
    # Normalize crop input
    crop = crop.lower().strip()
    if not 0 <= feedback_score <= 1:
        return "Error: Feedback score must be between 0 and 1"
    response = manage_irrigation(
        "Tifton", crop, agent, feedback=feedback_score
    )  # Default location for feedback
    output = (
        f"Feedback ({feedback_score}) applied for {crop}\n"
        f"Updated Recommendation: {response.responses}\n"
        f"Updated Rules: {agent.learned_rules[crop]}"
    )
    return output


def view_rules(crop):
    # Normalize crop input
    crop = crop.lower().strip()
    if crop in agent.learned_rules:
        return f"Current rules for {crop}: {agent.learned_rules[crop]}"
    supported_crops = ", ".join(
        key for key in agent.learned_rules.keys() if key in agent.default_rules
    )
    return f"Error: {crop} not recognized. Supported crops: {supported_crops}"


with gr.Blocks(title="Peanut Agentic AI System") as demo:
    gr.Markdown("# Peanut Agentic AI System")
    gr.Markdown(
        "This Agentic AI system provide a recommendation for irrigation based on current weather and soil conditions. "
    )

    # Tab 1: Check Irrigation
    with gr.Tab("Check Irrigation"):
        gr.Markdown("Enter details to see if your field needs irrigation.")
        crop_input = gr.Dropdown(
            choices=["cotton", "corn", "soybeans", "peanut"], label="Crop Type"
        )
        location_input = gr.Textbox(label="Location", placeholder="e.g., Tifton")
        lat_input = gr.Number(label="Latitude", value=31.4505)
        lon_input = gr.Number(label="Longitude", value=-83.5085)
        check_button = gr.Button("Get Recommendation")
        output_text = gr.Textbox(label="Irrigation Recommendation", lines=5)
        check_button.click(
            fn=check_irrigation,
            inputs=[crop_input, location_input, lat_input, lon_input],
            outputs=output_text,
        )

    # Tab 2: Provide Feedback
    with gr.Tab("Provide Feedback"):
        gr.Markdown("After irrigating, provide feedback (0 to 1) to help the AI learn.")
        feedback_crop = gr.Dropdown(
            choices=["cotton", "corn", "soybeans", "peanut"], label="Crop Type"
        )
        feedback_score = gr.Slider(
            minimum=0, maximum=1, step=0.1, label="Feedback Score (0 = bad, 1 = good)"
        )
        feedback_button = gr.Button("Submit Feedback")
        feedback_output = gr.Textbox(label="Feedback Result", lines=3)
        feedback_button.click(
            fn=provide_feedback,
            inputs=[feedback_crop, feedback_score],
            outputs=feedback_output,
        )

    # Tab 3: View Rules
    with gr.Tab("View Learned Rules"):
        gr.Markdown("See the current learned rules for each crop.")
        rules_crop = gr.Dropdown(
            choices=["cotton", "corn", "soybeans", "peanut"], label="Crop Type"
        )
        print(f"rules_crop:{rules_crop}")
        rules_button = gr.Button("Show Rules")
        rules_output = gr.Textbox(label="Learned Rules", lines=2)
        rules_button.click(fn=view_rules, inputs=rules_crop, outputs=rules_output)

if __name__ == "__main__":
    demo.launch()
