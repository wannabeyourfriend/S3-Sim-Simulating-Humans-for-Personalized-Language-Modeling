import unittest

from user_simulator.data import Persona, fmt_conversation
from user_simulator.prompts import render
import user_simulator.simulator as simulator


class TestSmokeBehaviorSamplerPrompt(unittest.TestCase):
    def test_render_full_behavior_sampler_prompt(self):
        persona = Persona(
            id="smoke_persona",
            summary="Product manager with moderate technical background, prefers concise practical answers.",
        )
        conversation = [
            {"role": "user", "content": "我在准备一个内部分享，想讲讲RAG落地怎么做。"},
            {"role": "assistant", "content": "可以从数据切分、召回、重排和评估四个部分搭框架。"},
            {"role": "user", "content": "先帮我明确下首版最小可行范围。"},
        ]
        previous_behaviors = [{"behavior": "Retrieval"}, {"behavior": "Clarification"}]
        previous_text = ", ".join(item["behavior"] for item in previous_behaviors)

        user_prompt = render(
            simulator._TMPL_CTRL_USER,
            profile_summary=persona.summary[:1200] or "N/A",
            conversation_history=fmt_conversation(conversation[-12:]) or "N/A",
            turn_number=3,
            total_turns=12,
            previous_behaviors=previous_text or "N/A",
            current_user_state="Intent: scope_definition; Emotion: mild urgency",
        )

        final_prompt = (
            "===== SYSTEM PROMPT =====\n"
            + simulator._CTRL_SYSTEM_RENDERED
            + "\n\n===== USER PROMPT =====\n"
            + user_prompt
        )
        print(final_prompt)

        self.assertIn("selected_behavior_index", final_prompt)
        self.assertNotIn("{profile_summary}", final_prompt)
        self.assertNotIn("{conversation_history}", final_prompt)
        self.assertNotIn("{turn_number}", final_prompt)
        self.assertNotIn("{total_turns}", final_prompt)
        self.assertNotIn("{previous_behaviors}", final_prompt)
        self.assertNotIn("{current_user_state}", final_prompt)


if __name__ == "__main__":
    unittest.main()
