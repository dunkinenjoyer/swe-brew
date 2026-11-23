#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(pwd)/.swe-brew-calculator-example}"
if [[ -e "$root" ]]; then
  printf 'Refusing to overwrite existing path: %s\n' "$root" >&2
  exit 1
fi

repo="$root/calculator"
case_dir="$root/case"
mkdir -p "$repo/src/main/java/example" "$repo/src/test/java/example" "$case_dir"

cat > "$repo/pom.xml" <<'EOF'
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>example</groupId>
  <artifactId>calculator</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <maven.compiler.release>17</maven.compiler.release>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>
  <dependencies>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>5.10.2</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.2.5</version>
      </plugin>
    </plugins>
  </build>
</project>
EOF

cat > "$repo/src/main/java/example/Calculator.java" <<'EOF'
package example;

public final class Calculator {
    public int add(int left, int right) {
        return left + right;
    }
}
EOF

cat > "$repo/src/test/java/example/CalculatorTest.java" <<'EOF'
package example;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void addsPositiveNumbers() {
        assertEquals(5, new Calculator().add(2, 3));
    }
}
EOF

git -C "$repo" init -q
git -C "$repo" add .
git -C "$repo" -c user.name='SWE-BREW example' -c user.email='example@example.invalid' commit -qm 'Base calculator'
base_commit="$(git -C "$repo" rev-parse HEAD)"

python3 - <<'PY' "$repo/src/main/java/example/Calculator.java" "$repo/src/test/java/example/CalculatorTest.java"
from pathlib import Path
import sys

source = Path(sys.argv[1])
source.write_text(source.read_text().replace(
    '        return left + right;\n',
    '        if (left < 0 || right < 0) {\n            return 0;\n        }\n        return left + right;\n',
), encoding='utf-8')

test = Path(sys.argv[2])
test.write_text(test.read_text().replace(
    '    }\n}\n',
    '    }\n\n    @Test\n    void returnsZeroForNegativeOperands() {\n        assertEquals(0, new Calculator().add(-1, 3));\n    }\n}\n',
), encoding='utf-8')
PY

git -C "$repo" add .
git -C "$repo" -c user.name='SWE-BREW example' -c user.email='example@example.invalid' commit -qm 'Return zero for negative operands'
head_commit="$(git -C "$repo" rev-parse HEAD)"
gold_patch="$(git -C "$repo" diff --binary "$base_commit" "$head_commit")"

REPO="$repo" BASE_COMMIT="$base_commit" HEAD_COMMIT="$head_commit" GOLD_PATCH="$gold_patch" CASE_DIR="$case_dir" python3 - <<'PY'
import json
import os
from pathlib import Path

case = {
    'case_id': 'calculator-negative-operands',
    'repository': os.environ['REPO'],
    'base_commit': os.environ['BASE_COMMIT'],
    'head_commit': os.environ['HEAD_COMMIT'],
    'problem_statement': 'Return zero from Calculator.add when either operand is negative. Preserve ordinary addition for non-negative operands.',
    'gold_patch': os.environ['GOLD_PATCH'],
    'changed_files': [
        'src/main/java/example/Calculator.java',
        'src/test/java/example/CalculatorTest.java',
    ],
    'test_commands': ['mvn -B -q test'],
}
Path(os.environ['CASE_DIR'], 'case.json').write_text(json.dumps(case, indent=2) + '\n', encoding='utf-8')
PY

printf 'Example case created at %s\n' "$case_dir"
printf 'Run: swe-brew-enhance %q --workspace %q --stub-llm\n' "$case_dir" "$root/workspace"
