from .controller import RoutedSkills


class SkillPromptExecutor:
    """Render the selected skill set as recommendation context."""

    @staticmethod
    def build_rec_prompt(
        base_prompt_template: str,
        routed_skills: RoutedSkills,
    ) -> str:
        skill_block = SkillPromptExecutor._render_skill_block(routed_skills)

        sections = [skill_block, base_prompt_template]
        sections.append(
            "Use only the skill signals that are applicable to the current user history "
            "and candidate set. Ignore weak, irrelevant, or conflicting skill guidance, "
            "and ground the final ranking in the current recommendation context."
        )

        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _render_skill_block(routed_skills: RoutedSkills) -> str:
        lines = ["## Selected Skill Guidance"]
        lines.append(f"Preference Query: {routed_skills.preference_query}")

        if routed_skills.user_skills:
            lines.append("### User-Profile Skills")
            for skill in routed_skills.user_skills:
                lines.append(skill.to_prompt())

        if routed_skills.functional_skills:
            lines.append("### Functional Skills")
            for skill in routed_skills.functional_skills:
                lines.append(skill.to_prompt())

        if not routed_skills.user_skills and not routed_skills.functional_skills:
            lines.append("No transferable skills were selected for this instance.")

        return "\n\n".join(lines)
