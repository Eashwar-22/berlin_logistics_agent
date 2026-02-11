import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from tools import (
    anonymize_pii,
    calculate_delivery_distance, 
    get_weather_risk, 
    predict_delivery_time,
    explain_delivery_prediction,
    check_data_drift,
    get_location_name)

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

tools = [anonymize_pii, 
        calculate_delivery_distance, 
        get_weather_risk, 
        predict_delivery_time,
        explain_delivery_prediction,
        check_data_drift,
        get_location_name]

agent = create_react_agent(model, tools)

def visualize_agent():
    try:
        graph_image = agent.get_graph().draw_mermaid_png()
        with open("agent_graph.png", "wb") as f:
            f.write(graph_image)
    except Exception as e:
        print(f"Could not draw graph: {e}")


def run_agent_logic(user_query: str):
    """
    Wraps the agent execution to make it easy to call.
    """
    print(f"Agent Processing: '{user_query}'")
    result = agent.invoke({"messages": [("user", user_query)]})
    
    # return last message
    return result["messages"][-1].content

if __name__ == "__main__":
    # visualize_agent()

    # query = "Calculate distance between 52.52, 13.40 and 52.50, 13.35. Then predict the delivery time for a Bike if the weather is Rainy."
    # query = "Predict delivery time for a Bike in Berlin when it is Rainy. Then explain why the prediction is that number."
    # query = "Calculate distance between 52.52, 13.40 and 52.50, 13.35. Then predict the delivery time for a Bike if the weather is Sunny. How will the duration be changed if I had used a different vehicle?"
    # query = "The delivery times today were [90, 85, 95, 88, 92]. Check if there is data drift."
    query = "Calculate distance between (52.52, 13.40) and (52.50, 13.35). Between which two areas in Berlin are these coordinates? Then predict delivery time for a Bike in High traffic with a Junior driver in Rain. Explain the prediction."
    
    print("\n--- Running Test ---")
    response = run_agent_logic(query)
    print(f"\nFinal Answer: {response}")