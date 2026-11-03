# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import AutoSciState
from agent.nodes.planner import planner_node
from agent.nodes.solver_reporter import solver_reporter_node
from agent.nodes.judge_reflector import judge_reflector_node


def route_next_node(state: AutoSciState):
    target = state.get("next_node", "end")
    return END if target == "end" else target


def build_autosci_graph():
    """
    Compiles the 3-node hierarchical closed-loop control system:
      1. planner         (Requirement + Pre-Audit Gate + Knowledge Lookup + Strategy)
      2. solver_reporter (Derive + Derivation Audit + Code & Debug + Execute & Verify + Diagnose + Report)
      3. judge_reflector (Post-Audit Scientific Verification + HARNESS² & RRSI Continual Experience Extraction)
    """
    workflow = StateGraph(AutoSciState)

    node_registry = [
        ("planner", planner_node),
        ("solver_reporter", solver_reporter_node),
        ("judge_reflector", judge_reflector_node),
    ]
    for name, fn in node_registry:
        workflow.add_node(name, fn)

    workflow.set_entry_point("planner")

    routing_map = {name: name for name, _ in node_registry}
    routing_map[END] = END

    for name, _ in node_registry:
        workflow.add_conditional_edges(name, route_next_node, routing_map)

    return workflow.compile()