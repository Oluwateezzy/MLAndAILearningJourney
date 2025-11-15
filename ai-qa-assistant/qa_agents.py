import json
from typing import Any, Dict, List

# from openai import OpenAI
from groq import Groq
from rag_system import DocumentProcessor, VectorStore

# client = OpenAI()
client = Groq()


class RetrieverAgent:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("user_query", "")
        feature_description = state.get("feature_description", "")

        search_query = f"{query} {feature_description}"
        relevant_docs = self.vector_store.retrieve_relevant_context(search_query)

        return {
            **state,
            "retrieved_context": relevant_docs,
            "agent_logs": state.get("agent_logs", [])
            + ["RetrieverAgent: Fetched relevant documentation"],
        }


class TestGeneratorAgent:

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        context = state.get("retrieved_context", [])
        feature_desc = state.get("feature_description", "")
        prompt = self._build_test_generation_prompt(context, feature_desc)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        test_cases = self._parse_test_cases(response.choices[0].message.content)

        return {
            **state,
            "generated_test_cases": test_cases,
            "agent_logs": state.get("agent_logs", [])
            + ["TestGeneratorAgent: Generated initial test cases"],
        }

    def _build_test_generation_prompt(
        self, context: List[str], feature_desc: str
    ) -> str:
        return f"""
        Based on the following context and feature description, generate comprehensive test cases.
        
        CONTEXT:
        {''.join(context)}
        
        FEATURE:
        {feature_desc}
        
        Generate test cases in this exact JSON format:
        {{
            "test_cases": [
                {{
                    "test_id": "TC001",
                    "description": "Clear description",
                    "preconditions": ["precondition1", "precondition2"],
                    "steps": ["step1", "step2", "step3"],
                    "expected_result": "Expected outcome",
                    "test_type": "positive|negative|edge",
                    "requirements_covered": ["req1", "req2"]
                }}
            ]
        }}
        
        Include positive, negative, and edge cases. Ensure each test case covers specific requirements.
        """

    def _parse_test_cases(self, content: str) -> List[Dict]:
        """Parse the model's response into a list of test cases."""
        try:
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            # Parse JSON content
            data = json.loads(content.strip())

            # Extract test cases
            if "test_cases" in data:
                return data["test_cases"]
            else:
                print("Warning: 'test_cases' key not found in response")
                return []

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print("Raw response content:")
            print(content)
            return []
        except Exception as e:
            print(f"Unexpected error parsing test cases: {e}")
            return []


if __name__ == "__main__":
    processor = DocumentProcessor()
    vector_store = VectorStore()

    # Step 1: Process a document
    docs = processor.process_document(["sample.txt"])

    # Step 2: Add document text into the vector store
    for i, doc in enumerate(docs):
        vector_store.add_documents(
            chunks=[doc["text"]], metadata=[{"source": f"doc_{i}"}]
        )

    # Step 3: Ask a question (query)
    query = "What is the capital of Nigeria?"

    results = vector_store.retrieve_relevant_context(query)

    retriever_agent = RetrieverAgent(vector_store)

    state = {
        "user_query": query,
        "feature_description": "The system should retrieve documents from the vector store and generate concise answers summarizing the information based on user queries. It must support positive, negative, and edge cases such as empty queries, irrelevant queries, multiple documents, and large documents.",
        "agent_logs": [],
    }

    results = retriever_agent(state)

    # Print the output
    print("Retrieved Context:")
    for doc in results["retrieved_context"]:
        print("-", doc)

    print("\nAgent Logs:")
    print(results["agent_logs"])

    test_agent = TestGeneratorAgent()
    state = test_agent(state)

    print("\n=== Generated Test Cases ===")
    for tc in state["generated_test_cases"]:
        print(tc)

    print("\n=== Final Logs ===")
    print(state["agent_logs"])
