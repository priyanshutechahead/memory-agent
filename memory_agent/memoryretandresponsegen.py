def generate_answer_from_memories(self, query: str, memories: List[Dict[str, Any]]) -> str:
    # Format memories into a string for the LLM
    memories_str = "\n".join([f"- {mem['memory']}" for mem in memories])

    # Create a prompt that includes user context
    prompt = f"""
User ID: {self.user_id}
User question: "{query}"

Relevant memories for user {self.user_id}:
{memories_str}

Please generate a helpful response using only the memories related to the question.
Try to answer to the point.
"""

    # Use the LLM to generate a response based on memories
    response = self.agent.tool.use_llm(
        prompt=prompt,
        system_prompt=ANSWER_SYSTEM_PROMPT
    )

    return str(response['content'][0]['text'])