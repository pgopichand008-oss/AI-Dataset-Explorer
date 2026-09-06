import time
import os
from google import genai


# ============================================================
# GEMINI MODEL CONFIGURATION
# ============================================================

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

MAX_RETRIES = 2
INITIAL_DELAY = 2


# ============================================================
# GEMINI CLIENT
# ============================================================

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


# ============================================================
# SAFE TEXT
# ============================================================

def _safe_text(value):

    if value is None:
        return "None"

    try:
        return str(value)
    except Exception:
        return "Unavailable"


# ============================================================
# COMPACT DATAFRAME
# ============================================================

def _compact_dataframe(
    data,
    max_rows=20,
    max_chars=8000
):

    if data is None:
        return ""

    try:

        if hasattr(data, "head"):
            text = data.head(max_rows).to_string()
        else:
            text = str(data)

        return text[:max_chars]

    except Exception:

        return _safe_text(data)[:max_chars]


# ============================================================
# BUILD EVIDENCE
# ============================================================

def _build_evidence(
    profile=None,
    quality_findings=None,
    agent_decisions=None,
    executive_summary=None,
    numerical_summary=None,
    categorical_summary=None,
    visualization_plan=None,
    quality_score=None,
    ml_readiness=None,
):

    evidence = []

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    if profile:

        evidence.append(
            "DATASET PROFILE:\n"
            + _safe_text(profile)
        )

    # --------------------------------------------------------
    # QUALITY SCORE
    # --------------------------------------------------------

    if quality_score is not None:

        evidence.append(
            f"DATA QUALITY SCORE: {quality_score}/100"
        )

    # --------------------------------------------------------
    # ML READINESS
    # --------------------------------------------------------

    if ml_readiness is not None:

        evidence.append(
            f"ML READINESS SCORE: {ml_readiness}/100"
        )

    # --------------------------------------------------------
    # QUALITY FINDINGS
    # --------------------------------------------------------

    if quality_findings:

        evidence.append(
            "QUALITY FINDINGS:\n"
            + _safe_text(quality_findings)
        )

    # --------------------------------------------------------
    # AGENT DECISIONS
    # --------------------------------------------------------

    if agent_decisions:

        evidence.append(
            "AGENT DECISIONS:\n"
            + _safe_text(agent_decisions)
        )

    # --------------------------------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------------------------------

    if executive_summary:

        evidence.append(
            "EXECUTIVE SUMMARY:\n"
            + _safe_text(executive_summary)
        )

    # --------------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------------

    if numerical_summary is not None:

        compact = _compact_dataframe(
            numerical_summary,
            max_rows=20,
            max_chars=8000
        )

        if compact:

            evidence.append(
                "NUMERICAL SUMMARY:\n"
                + compact
            )

    # --------------------------------------------------------
    # CATEGORICAL SUMMARY
    # --------------------------------------------------------

    if categorical_summary is not None:

        compact = _compact_dataframe(
            categorical_summary,
            max_rows=20,
            max_chars=8000
        )

        if compact:

            evidence.append(
                "CATEGORICAL SUMMARY:\n"
                + compact
            )

    # --------------------------------------------------------
    # VISUALIZATION PLAN
    # --------------------------------------------------------

    if visualization_plan:

        evidence.append(
            "VISUALIZATION PLAN:\n"
            + _safe_text(visualization_plan)
        )

    return "\n\n".join(evidence)


# ============================================================
# GEMINI PROMPT
# ============================================================

def _build_prompt(evidence):

    return f"""
You are an expert AI Dataset Intelligence Analyst.

You are part of a data-analysis application that first performs
statistical and data-quality analysis using Python and then asks
you to interpret the analytical evidence.

Your responsibility is to transform the supplied evidence into
a professional executive-level intelligence report.

IMPORTANT RULES:

- Use ONLY the supplied analytical evidence.
- Never invent statistics.
- Never invent column names.
- Never invent relationships between variables.
- Do not claim causation from correlation.
- Do not state that a column is definitely a target unless the
  evidence clearly supports that conclusion.
- Clearly distinguish evidence from interpretation.
- Mention uncertainty where appropriate.
- Do not repeat large tables unnecessarily.
- Keep the report detailed enough for a technical judge but
  concise enough to read quickly.

Create the report using EXACTLY these sections:

# Executive Assessment

Write 2–3 concise paragraphs describing the overall condition
of the dataset, its strengths, important risks, and whether it
appears suitable for further analysis.

# Dataset Overview

Summarize the important dataset characteristics including
size, columns, missing values, duplicates, quality score,
and ML-readiness score when available.

# Key Findings

Provide 5–7 important findings.

Prioritize findings based on analytical importance rather than
simply listing every observation.

# Data Quality Assessment

Explain the most important data-quality issues such as:

- Missing values
- Duplicate records
- Outliers
- Constant columns
- High-cardinality columns
- Potential identifiers
- Other limitations supported by the evidence

Explain why the issues matter.

# Statistical Insights

Discuss important statistical patterns supported by the
numerical and categorical summaries.

Consider:

- Distribution characteristics
- Skewness
- Range
- Dominant categories
- Potential relationships
- Important variation

Do not invent relationships that are not present in the evidence.

# ML Readiness

Assess whether the dataset appears suitable for machine-learning
analysis.

Discuss:

- Possible target variables
- Feature suitability
- Missing data
- Outliers
- Identifiers
- Data quality risks
- Any preprocessing that may be required

Use cautious language.

# Opportunities

Identify 3–5 useful opportunities for further analysis.

Examples include:

- Potential predictive modeling
- Feature engineering
- Deeper statistical analysis
- Time-series analysis
- Segmentation
- Data-quality improvement

Only recommend opportunities supported by the evidence.

# Risks and Limitations

Clearly mention the main limitations and risks.

Do not exaggerate.

# Recommended Action

Give a practical sequence of next steps.

Use a numbered list.

# Final Verdict

End with ONE of these categories when sufficient evidence
supports it:

🟢 READY FOR FURTHER ANALYSIS

🟡 NEEDS ATTENTION BEFORE ANALYSIS

🔴 NOT SUITABLE WITHOUT ADDITIONAL CORRECTION

Then give a short explanation for the verdict.

ANALYTICAL EVIDENCE:

{evidence}
"""


# ============================================================
# TEMPORARY ERROR DETECTION
# ============================================================

def _is_temporary_error(error):

    message = str(error).lower()

    temporary_keywords = [
        "503",
        "unavailable",
        "high demand",
        "429",
        "rate limit",
        "resource exhausted",
        "timeout",
        "timed out",
        "temporarily",
        "internal server error",
        "server error",
    ]

    return any(
        keyword in message
        for keyword in temporary_keywords
    )


# ============================================================
# GENERATE WITH ONE MODEL
# ============================================================

def _generate_with_model(
    client,
    model,
    prompt
):

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            text = getattr(
                response,
                "text",
                None
            )

            if text and str(text).strip():

                return (
                    str(text).strip(),
                    None
                )

            last_error = (
                "Gemini returned an empty response."
            )

        except Exception as error:

            last_error = error

            if not _is_temporary_error(error):
                break

            if attempt < MAX_RETRIES - 1:

                delay = INITIAL_DELAY * (
                    2 ** attempt
                )

                time.sleep(delay)

    return None, last_error


# ============================================================
# LOCAL FALLBACK REPORT
# ============================================================

def _local_fallback_report(
    profile=None,
    quality_findings=None,
    agent_decisions=None,
    executive_summary=None,
    quality_score=None,
    ml_readiness=None,
):

    lines = []

    # --------------------------------------------------------
    # EXECUTIVE ASSESSMENT
    # --------------------------------------------------------

    lines.append("# Executive Assessment")

    lines.append(
        "The dataset was successfully analyzed by the local "
        "Dataset Intelligence Engine. Gemini's natural-language "
        "interpretation is temporarily unavailable, so this "
        "report is based on the analytical evidence already "
        "generated by the application."
    )

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    lines.append("")
    lines.append("# Dataset Overview")

    if isinstance(profile, dict):

        if "rows" in profile:
            lines.append(
                f"- Rows: {profile['rows']}"
            )

        if "columns" in profile:
            lines.append(
                f"- Columns: {profile['columns']}"
            )

        if "missing_cells" in profile:
            lines.append(
                f"- Missing cells: "
                f"{profile['missing_cells']}"
            )

        if "duplicate_rows" in profile:
            lines.append(
                f"- Duplicate rows: "
                f"{profile['duplicate_rows']}"
            )

    if quality_score is not None:

        lines.append(
            f"- Data Quality Score: "
            f"{quality_score}/100"
        )

    if ml_readiness is not None:

        lines.append(
            f"- ML Readiness Score: "
            f"{ml_readiness}/100"
        )

    # --------------------------------------------------------
    # KEY FINDINGS
    # --------------------------------------------------------

    lines.append("")
    lines.append("# Key Findings")

    if isinstance(
        quality_findings,
        list
    ):

        for finding in quality_findings[:7]:

            if isinstance(finding, dict):

                finding_text = finding.get(
                    "Finding",
                    finding.get(
                        "finding",
                        str(finding)
                    )
                )

                lines.append(
                    f"- {_safe_text(finding_text)}"
                )

            else:

                lines.append(
                    f"- {_safe_text(finding)}"
                )

    elif isinstance(
        quality_findings,
        dict
    ):

        for key, value in list(
            quality_findings.items()
        )[:7]:

            lines.append(
                f"- {key}: "
                f"{_safe_text(value)}"
            )

    elif quality_findings:

        text = _safe_text(
            quality_findings
        )

        for line in text.splitlines()[:7]:

            lines.append(
                f"- {line}"
            )

    else:

        lines.append(
            "- No major quality findings were supplied."
        )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    lines.append("")
    lines.append("# Data Quality Assessment")

    if quality_findings:

        lines.append(
            _safe_text(quality_findings)
        )

    else:

        lines.append(
            "No additional quality findings were supplied."
        )

    # --------------------------------------------------------
    # ML READINESS
    # --------------------------------------------------------

    lines.append("")
    lines.append("# ML Readiness")

    if ml_readiness is not None:

        if ml_readiness >= 80:

            readiness_text = (
                "The dataset appears mostly ready for "
                "further machine-learning analysis, subject "
                "to normal preprocessing and validation."
            )

        elif ml_readiness >= 60:

            readiness_text = (
                "The dataset may be usable for machine-learning "
                "analysis, but several preprocessing or quality "
                "issues should be reviewed first."
            )

        else:

            readiness_text = (
                "The dataset requires additional preparation "
                "before machine-learning analysis."
            )

        lines.append(
            readiness_text
        )

    else:

        lines.append(
            "ML readiness score was not supplied."
        )

    # --------------------------------------------------------
    # OPPORTUNITIES
    # --------------------------------------------------------

    lines.append("")
    lines.append("# Opportunities")

    lines.append(
        "Further opportunities should be selected based on "
        "the available column intelligence, statistical "
        "summaries, and visualization analysis."
    )

    # --------------------------------------------------------
    # RISKS
    # --------------------------------------------------------

    lines.append("")
    lines.append("# Risks and Limitations")

    lines.append(
        "Gemini-based interpretation is temporarily unavailable. "
        "The local analytical results should therefore be "
        "reviewed before making high-impact decisions."
    )

    # --------------------------------------------------------
    # RECOMMENDED ACTION
    # --------------------------------------------------------

    lines.append("")
    lines.append("# Recommended Action")

    lines.append(
        "1. Review the identified data-quality findings."
    )

    lines.append(
        "2. Address important missing values or data-quality issues."
    )

    lines.append(
        "3. Validate potential target variables and features."
    )

    lines.append(
        "4. Perform deeper statistical or ML analysis after validation."
    )

    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    lines.append("")
    lines.append("# Final Verdict")

    if ml_readiness is not None:

        if ml_readiness >= 80:

            verdict = (
                "🟢 READY FOR FURTHER ANALYSIS"
            )

        elif ml_readiness >= 60:

            verdict = (
                "🟡 NEEDS ATTENTION BEFORE ANALYSIS"
            )

        else:

            verdict = (
                "🔴 NOT SUITABLE WITHOUT ADDITIONAL CORRECTION"
            )

        lines.append(verdict)

    else:

        lines.append(
            "🟡 NEEDS ATTENTION BEFORE ANALYSIS"
        )

    # --------------------------------------------------------
    # AI STATUS
    # --------------------------------------------------------

    lines.append("")
    lines.append("# AI Status")

    lines.append(
        "Gemini is temporarily unavailable. "
        "The local Dataset Intelligence Engine completed "
        "the underlying analysis successfully."
    )

    return "\n".join(lines)


# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_ai_report(
    profile=None,
    quality_findings=None,
    agent_decisions=None,
    executive_summary=None,
    numerical_summary=None,
    categorical_summary=None,
    visualization_plan=None,
    quality_score=None,
    ml_readiness=None,
    **kwargs,
):

    """
    Generate a medium-detail executive AI report.

    The function is intentionally compatible with the current
    application and accepts additional keyword arguments.
    """

    client = _get_client()

    # --------------------------------------------------------
    # NO API KEY
    # --------------------------------------------------------

    if client is None:

        return _local_fallback_report(
            profile=profile,
            quality_findings=quality_findings,
            agent_decisions=agent_decisions,
            executive_summary=executive_summary,
            quality_score=quality_score,
            ml_readiness=ml_readiness,
        )

    # --------------------------------------------------------
    # BUILD EVIDENCE
    # --------------------------------------------------------

    evidence = _build_evidence(
        profile=profile,
        quality_findings=quality_findings,
        agent_decisions=agent_decisions,
        executive_summary=executive_summary,
        numerical_summary=numerical_summary,
        categorical_summary=categorical_summary,
        visualization_plan=visualization_plan,
        quality_score=quality_score,
        ml_readiness=ml_readiness,
    )

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = _build_prompt(
        evidence
    )

    errors = []

    # --------------------------------------------------------
    # TRY EACH MODEL
    # --------------------------------------------------------

    for model in MODELS:

        text, error = _generate_with_model(
            client,
            model,
            prompt
        )

        if text:

            return text

        if error:

            errors.append(
                f"{model}: {error}"
            )

    # --------------------------------------------------------
    # GEMINI FAILED
    # USE LOCAL FALLBACK
    # --------------------------------------------------------

    fallback_report = _local_fallback_report(
        profile=profile,
        quality_findings=quality_findings,
        agent_decisions=agent_decisions,
        executive_summary=executive_summary,
        quality_score=quality_score,
        ml_readiness=ml_readiness,
    )

    diagnostics = ""

    if errors:

        diagnostics = (
            "\n\n---\n\n"
            "Gemini service diagnostics:\n"
            + "\n".join(errors[:5])
        )

    return (
        fallback_report
        + diagnostics
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def generate_executive_report(
    profile=None,
    quality_findings=None,
    agent_decisions=None,
    executive_summary=None,
    numerical_summary=None,
    categorical_summary=None,
    visualization_plan=None,
    quality_score=None,
    ml_readiness=None,
    **kwargs,
):

    return generate_ai_report(
        profile=profile,
        quality_findings=quality_findings,
        agent_decisions=agent_decisions,
        executive_summary=executive_summary,
        numerical_summary=numerical_summary,
        categorical_summary=categorical_summary,
        visualization_plan=visualization_plan,
        quality_score=quality_score,
        ml_readiness=ml_readiness,
        **kwargs,
    )