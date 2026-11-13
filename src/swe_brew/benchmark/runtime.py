"""Runtime capability specification.

This is the local analogue of RepoLaunch's language/platform handlers:
the pipeline should consume a small set of executable commands and file
conventions from a runtime spec, not sprinkle language-specific command
strings through every module.

Defaults remain Java/Maven because that is the current corpus. Operators
can override the executable commands with environment variables; adding a
new runtime kind should be a config change, not a pipeline rewrite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSpec:
    """Commands + file conventions used by setup, organize, and verify."""

    kind: str
    base_image: str
    test_command: str
    compile_command: str
    install_command: str
    pertest_template: str
    java_tool_options: str
    command_timeout_seconds: int
    trial_timeout_seconds: int
    test_path_marker: str = "/src/test/"
    test_file_suffixes: tuple[str, ...] = ("Test.java", "Tests.java", "IT.java", "ITCase.java")
    report_format: str = "surefire"


_JAVA_MAVEN = RuntimeSpec(
    kind="java-maven",
    base_image="eclipse-temurin:17-jdk-jammy",
    test_command=(
        "mvn -B -q "
        "-Dmaven.test.failure.ignore=true "
        "-Dmaven.compiler.failOnError=false "
        "-Dmaven.javadoc.skip=true "
        "test"
    ),
    compile_command=(
        "mvn -B -q -e "
        "-Dmaven.compiler.failOnError=false "
        "-Dmaven.javadoc.skip=true "
        "test-compile"
    ),
    install_command=(
        "mvn -B "
        "-Dmaven.compiler.failOnError=false "
        "-Dmaven.javadoc.skip=true "
        "install -DskipTests"
    ),
    pertest_template=(
        "mvn -B -q -pl {module} -Dtest={cls}#{method} test"
    ),
    java_tool_options=(
        "--add-opens jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED "
        "--add-opens jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED "
        "--add-opens jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED "
        "--add-opens jdk.compiler/com.sun.tools.javac.main=ALL-UNNAMED "
        "--add-opens jdk.compiler/com.sun.tools.javac.model=ALL-UNNAMED "
        "--add-opens jdk.compiler/com.sun.tools.javac.parser=ALL-UNNAMED "
        "--add-opens jdk.compiler/com.sun.tools.javac.processing=ALL-UNNAMED "
        "--add-opens jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED "
        "--add-opens jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED "
        "--add-opens jdk.compiler/com.sun.tools.javac.jvm=ALL-UNNAMED"
    ),
    command_timeout_seconds=3600,
    trial_timeout_seconds=7200,
    test_path_marker="/src/test/java/",
    test_file_suffixes=("Test.java", "Tests.java", "IT.java", "ITCase.java"),
    report_format="surefire",
)


def runtime_spec() -> RuntimeSpec:
    """Return the configured runtime spec.

    Environment overrides are intentionally generic command slots, matching
    RepoLaunch's base-image + commands guidance rather than a Java-specific
    flag convention.
    """
    kind = os.environ.get("SWE_BREW_RUNTIME_KIND", "java-maven").strip().lower()
    if kind == "java-maven":
        spec = _JAVA_MAVEN
    else:
        spec = RuntimeSpec(
            kind=kind,
            base_image=os.environ.get("SWE_BREW_BASE_IMAGE", "eclipse-temurin:17-jdk-jammy"),
            test_command=os.environ.get("SWE_BREW_TEST_COMMAND", ""),
            compile_command=os.environ.get("SWE_BREW_COMPILE_COMMAND", ""),
            install_command=os.environ.get("SWE_BREW_INSTALL_COMMAND", ""),
            pertest_template=os.environ.get(
                "SWE_BREW_PERTEST_TEMPLATE",
                "mvn -B -q -pl {module} -Dtest={cls}#{method} test",
            ),
            java_tool_options=os.environ.get(
                "SWE_BREW_JAVA_TOOL_OPTIONS",
                "",
            ),
            command_timeout_seconds=int(os.environ.get("SWE_BREW_CMD_TIMEOUT", "3600")),
            trial_timeout_seconds=int(os.environ.get("SWE_BREW_TRIAL_TIMEOUT", "7200")),
            test_path_marker=os.environ.get("SWE_BREW_TEST_PATH_MARKER", "/src/test/"),
            test_file_suffixes=tuple(
                s.strip()
                for s in os.environ.get(
                    "SWE_BREW_TEST_FILE_SUFFIXES",
                    "Test.java,Tests.java,IT.java,ITCase.java",
                ).split(",")
                if s.strip()
            ),
            report_format=os.environ.get("SWE_BREW_REPORT_FORMAT", "surefire"),
        )

    if os.environ.get("SWE_BREW_TEST_COMMAND"):
        spec = RuntimeSpec(
            kind=spec.kind,
            base_image=os.environ.get("SWE_BREW_BASE_IMAGE", spec.base_image),
            test_command=os.environ["SWE_BREW_TEST_COMMAND"],
            compile_command=os.environ.get("SWE_BREW_COMPILE_COMMAND", spec.compile_command),
            install_command=os.environ.get("SWE_BREW_INSTALL_COMMAND", spec.install_command),
            pertest_template=os.environ.get("SWE_BREW_PERTEST_TEMPLATE", spec.pertest_template),
            java_tool_options=os.environ.get("SWE_BREW_JAVA_TOOL_OPTIONS", spec.java_tool_options),
            command_timeout_seconds=int(
                os.environ.get("SWE_BREW_CMD_TIMEOUT", str(spec.command_timeout_seconds))
            ),
            trial_timeout_seconds=int(
                os.environ.get("SWE_BREW_TRIAL_TIMEOUT", str(spec.trial_timeout_seconds))
            ),
            test_path_marker=os.environ.get("SWE_BREW_TEST_PATH_MARKER", spec.test_path_marker),
            test_file_suffixes=spec.test_file_suffixes,
            report_format=os.environ.get("SWE_BREW_REPORT_FORMAT", spec.report_format),
        )
    return spec
