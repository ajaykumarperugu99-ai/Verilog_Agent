import os
import logging
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verilog_agent")

# --- 1. Define Tools: each returns a verified Verilog module ---
@tool
def generate_half_adder(circuit_name: str = "half_adder") -> str:
    """Generate Verilog code for a Half Adder (1-bit sum + carry, no carry-in).
    circuit_name: optional label, leave as default 'half_adder'.
    """
    return (
        "module half_adder(\n"
        "    input  a,\n"
        "    input  b,\n"
        "    output sum,\n"
        "    output carry\n"
        ");\n"
        "    assign sum   = a ^ b;\n"
        "    assign carry = a & b;\n"
        "endmodule"
    )

@tool
def generate_full_adder(circuit_name: str = "full_adder") -> str:
    """Generate Verilog code for a Full Adder (1-bit sum + carry-out, with carry-in).
    circuit_name: optional label, leave as default 'full_adder'.
    """
    return (
        "module full_adder(\n"
        "    input  a,\n"
        "    input  b,\n"
        "    input  cin,\n"
        "    output sum,\n"
        "    output cout\n"
        ");\n"
        "    assign sum  = a ^ b ^ cin;\n"
        "    assign cout = (a & b) | (b & cin) | (a & cin);\n"
        "endmodule"
    )

@tool
def generate_sr_flip_flop(circuit_name: str = "sr_flip_flop") -> str:
    """Generate Verilog code for a clocked SR (Set-Reset) Flip-Flop.
    circuit_name: optional label, leave as default 'sr_flip_flop'.
    """
    return (
        "module sr_flip_flop(\n"
        "    input      clk,\n"
        "    input      s,\n"
        "    input      r,\n"
        "    output reg q,\n"
        "    output reg qbar\n"
        ");\n"
        "    always @(posedge clk) begin\n"
        "        case ({s, r})\n"
        "            2'b00: q <= q;      // hold\n"
        "            2'b01: q <= 1'b0;   // reset\n"
        "            2'b10: q <= 1'b1;   // set\n"
        "            2'b11: q <= 1'bx;   // invalid state\n"
        "        endcase\n"
        "        qbar <= ~q;\n"
        "    end\n"
        "endmodule"
    )

@tool
def generate_d_flip_flop(circuit_name: str = "d_flip_flop") -> str:
    """Generate Verilog code for a D (Data) Flip-Flop with asynchronous reset.
    circuit_name: optional label, leave as default 'd_flip_flop'.
    """
    return (
        "module d_flip_flop(\n"
        "    input      clk,\n"
        "    input      rst,\n"
        "    input      d,\n"
        "    output reg q\n"
        ");\n"
        "    always @(posedge clk or posedge rst) begin\n"
        "        if (rst)\n"
        "            q <= 1'b0;\n"
        "        else\n"
        "            q <= d;\n"
        "    end\n"
        "endmodule"
    )

@tool
def generate_logic_gate(gate: str) -> str:
    """Generate Verilog code for a basic logic gate.
    gate must be one of: and, or, nand, nor, xor, xnor, not.
    """
    gate = gate.strip().lower()
    exprs = {
        "and":  "y = a & b",
        "or":   "y = a | b",
        "nand": "y = ~(a & b)",
        "nor":  "y = ~(a | b)",
        "xor":  "y = a ^ b",
        "xnor": "y = ~(a ^ b)",
        "not":  "y = ~a",
    }
    if gate not in exprs:
        return f"Unknown gate '{gate}'. Supported gates: {', '.join(exprs)}."
    if gate == "not":
        ports = "input  a,\n    output y"
    else:
        ports = "input  a,\n    input  b,\n    output y"
    return (
        f"module {gate}_gate(\n    {ports}\n);\n"
        f"    assign {exprs[gate]};\n"
        "endmodule"
    )

tools = [generate_half_adder, generate_full_adder, generate_sr_flip_flop,
         generate_d_flip_flop, generate_logic_gate]

# --- 2. Initialize Model & Agent ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-3-1b-it")

llm_flash = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    api_key=GOOGLE_API_KEY,
    temperature=0
)

VERILOG_SYSTEM_PROMPT = (
    "You are a specialized agent restricted ONLY to generating Verilog HDL code for basic digital logic "
    "circuits: half adder, full adder, SR flip-flop, D flip-flop, and the logic gates AND, OR, NAND, NOR, "
    "XOR, XNOR and NOT. Always use the provided tools to produce the Verilog code rather than writing it "
    "yourself, and present the returned code inside a ```verilog code block. "
    "For any other roles, topics, or questions outside of Verilog digital logic design, you must say exactly: "
    "'I am not authorized to answer questions outside of Verilog digital logic design.'"
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=VERILOG_SYSTEM_PROMPT,
)

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    logger.info("Agent input: %r", user_input)
    return {"messages": [("user", user_input)]}

def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    # Case 1: top-level messages (normal final state)
    messages = agent_output.get("messages")

    # Case 2: nested under a node name, e.g. {"model": {"messages": [...]}}
    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))
        if not content:
            return (
                "[Agent returned an empty response. This usually means the model call "
                "returned zero generations. Check the Render logs for the raw "
                "ChatGoogleGenerativeAI output, and confirm GEMINI_API_KEY and MODEL_NAME "
                "are set correctly.]"
            )
        return content

    return f"[No messages returned. Raw agent output: {agent_output}]"

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

# --- 3. FastAPI App ---
app = FastAPI(title="Verilog Code Generation Agent")

@app.get("/health")
def health():
    return {"status": "ok"}

add_routes(app, formatted_agent_chain, path="/agent")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
