import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables (MEM0_API_KEY, GOOGLE_API_KEY, etc.)
load_dotenv()

try:
    import ollama
except ImportError:
    ollama = None

try:
    from mem0 import Memory
except ImportError:
    Memory = None

class MemoryAssistant:
    def __init__(self, user_id: str = "demo_user", mem0_config: Dict[str, Any] = None):
        self.user_id = user_id
        
        # Initialize the mem0 client (requires 'mem0ai' package)
        if Memory:
            if not mem0_config:
                mem0_config = {
                    "llm": {
                        "provider": "ollama",
                        "config": {"model": "gemma3:12b"}
                    },
                    "embedder": {
                        "provider": "ollama",
                        "config": {"model": "qwen3-embedding:0.6b"}
                    }
                }
            self.mem0_client = Memory.from_config(mem0_config) if mem0_config else Memory()
        else:
            self.mem0_client = None
            print("Warning: 'mem0ai' package is not installed.")

        # In a real setup, instantiate your LLM/Agent client here:
        # self.agent = Agent(
        #     system_prompt=MEMORY_SYSTEM_PROMPT,
        #     tools=[mem0_memory, use_llm],
        # )

    def process_input(self, user_input: str) -> str:
        # 1. Check if this is a memory storage request
        if user_input.lower().startswith(("remember ", "note that ", "i want you to know ")):
            content = user_input.split(" ", 1)[1]
            self.store_memory(content)
            return "I've stored that information in my memory."

        # 2. Check if this is a request to list all memories
        if "show" in user_input.lower() and "memories" in user_input.lower():
            all_memories = self.list_all_memories()
            if not all_memories:
                return "You don't have any stored memories yet."
            # Properly format and return the memory list, preventing fall-through
            formatted_memories = "\n".join([f"- {m.get('text', m.get('content'))}" for m in all_memories])
            return f"Here are your stored memories:\n{formatted_memories}"

        # 3. Otherwise, perform semantic retrieval and generate grounded response
        relevant_memories = self.retrieve_memories(user_input)
        return self.generate_answer_from_memories(user_input, relevant_memories)

    def store_memory(self, content: str) -> Dict[str, Any]:
        if not self.mem0_client:
            raise RuntimeError("mem0_client not initialized. Please install 'mem0ai' package.")
        
        # Add the memory using the mem0 client
        result = self.mem0_client.add(content, user_id=self.user_id)
        return result

    def _normalize_results(self, results: Any) -> List[Dict[str, Any]]:
        if not results:
            return []
        
        # Handle cases where mem0 returns an error dictionary or nested results
        if isinstance(results, dict):
            if "results" in results:
                results = results["results"]
            elif "memories" in results:
                results = results["memories"]
            elif "error" in results or "message" in results:
                print(f"Mem0 API returned an error: {results}")
                return []
            else:
                results = [results]
                
        if isinstance(results, str):
            print(f"Mem0 API returned a string: {results}")
            return []
            
        normalized = []
        for r in results:
            if isinstance(r, dict):
                normalized.append(r)
            elif hasattr(r, 'model_dump'):
                normalized.append(r.model_dump())
            elif hasattr(r, '__dict__'):
                normalized.append(r.__dict__)
            else:
                normalized.append({"memory": str(r)})
                
        return normalized

    def retrieve_memories(self, query: str, min_score: float = 0.3, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.mem0_client:
            raise RuntimeError("mem0_client not initialized.")
            
        try:
            results = self.mem0_client.search(query, filters={"user_id": self.user_id}, limit=max_results)
        except Exception as e:
            print(f"Search error: {e}")
            return []
            
        normalized = self._normalize_results(results)
        filtered_results = [r for r in normalized if r.get("score", 1.0) >= min_score]
        return filtered_results

    def list_all_memories(self) -> List[Dict[str, Any]]:
        if not self.mem0_client:
            raise RuntimeError("mem0_client not initialized.")
            
        try:
            results = self.mem0_client.get_all(filters={"user_id": self.user_id})
        except Exception as e:
            print(f"Get all error: {e}")
            return []
            
        return self._normalize_results(results)

    def generate_answer_from_memories(self, query: str, memories: List[Dict[str, Any]]) -> str:
        # Format memories into a string for the LLM
        memories_str = "\n".join([f"- {mem.get('memory', mem.get('text', ''))}" for mem in memories])

        # Create a prompt that includes user context
        prompt = f"""
User ID: {self.user_id}
User question: "{query}"

Relevant memories for user {self.user_id}:
{memories_str}

Please generate a helpful response using only the memories related to the question.
Try to answer to the point.
"""

        # Use the local Ollama LLM to generate a response based on memories
        if not ollama:
            return "Error: 'ollama' Python package not installed."
            
        try:
            response = ollama.chat(model='gemma3:12b', messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful memory assistant. Answer based on the provided memories. Be concise and to the point.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error generating response from local Llama: {str(e)}"

if __name__ == "__main__":
    print("Initializing Memory Assistant...")
    try:
        assistant = MemoryAssistant(user_id="demo_user")
        print("\nMemory Assistant is ready! (Type 'exit' to stop)")
        print("Try commands like:")
        print("  - 'Remember my favorite color is blue'")
        print("  - 'Show my memories'")
        print("  - 'What is my favorite color?'\n")
        
        while True:
            try:
                user_input = input("You: ")
            except (KeyboardInterrupt, EOFError):
                break
                
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
                
            response = assistant.process_input(user_input)
            print(f"Assistant: {response}\n")
            
    except Exception as e:
        print(f"\nFailed to start: {e}")