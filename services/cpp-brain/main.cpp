#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string lowerUtf8(std::string value) {
    for (size_t i = 0; i < value.size(); ++i) {
        unsigned char u = static_cast<unsigned char>(value[i]);
        if (u >= 'A' && u <= 'Z') {
            value[i] = static_cast<char>(u - 'A' + 'a');
            continue;
        }
        if (u == 0xD0 && i + 1 < value.size()) {
            unsigned char next = static_cast<unsigned char>(value[i + 1]);
            if (next == 0x81) {
                value[i] = static_cast<char>(0xD1);
                value[i + 1] = static_cast<char>(0x91);
                ++i;
            } else if (next >= 0x90 && next <= 0x9F) {
                value[i + 1] = static_cast<char>(next + 0x20);
                ++i;
            } else if (next >= 0xA0 && next <= 0xAF) {
                value[i] = static_cast<char>(0xD1);
                value[i + 1] = static_cast<char>(next - 0x20);
                ++i;
            }
        }
    }
    return value;
}

static std::string trim(std::string s) {
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static bool contains(const std::string &s, const std::string &part) {
    return s.find(part) != std::string::npos;
}

static bool containsAny(const std::string &s, const std::vector<std::string> &parts) {
    for (const auto &part : parts) if (contains(s, part)) return true;
    return false;
}

static std::string jsonEscape(const std::string &s) {
    std::ostringstream out;
    for (unsigned char ch : s) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    const char *hex = "0123456789abcdef";
                    out << "\\u00" << hex[ch >> 4] << hex[ch & 0x0f];
                } else out << static_cast<char>(ch);
        }
    }
    return out.str();
}

class ArithmeticParser {
    const std::string &s;
    size_t p = 0;
    void spaces() { while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p; }
    bool take(char c) { spaces(); if (p < s.size() && s[p] == c) { ++p; return true; } return false; }
    double primary() {
        spaces();
        if (take('(')) {
            double v = expression();
            if (!take(')')) throw std::runtime_error("paren");
            return v;
        }
        spaces();
        const size_t start = p;
        while (p < s.size() && ((s[p] >= '0' && s[p] <= '9') || s[p] == '.')) ++p;
        if (p == start) throw std::runtime_error("number");
        char *end = nullptr;
        const double value = std::strtod(s.substr(start, p - start).c_str(), &end);
        if (!std::isfinite(value) || std::abs(value) > 1e100) throw std::runtime_error("range");
        return value;
    }
    double power() {
        double a = primary();
        if (take('^')) {
            double b = unary();
            if (std::abs(b) > 100) throw std::runtime_error("exponent");
            const double v = std::pow(a, b);
            if (!std::isfinite(v) || std::abs(v) > 1e100) throw std::runtime_error("range");
            return v;
        }
        return a;
    }
    double unary() {
        if (take('+')) return unary();
        if (take('-')) return -unary();
        return power();
    }
    double term() {
        double v = unary();
        for (;;) {
            if (take('*')) v *= unary();
            else if (take('/')) {
                const double b = unary();
                if (b == 0) throw std::runtime_error("zero");
                v /= b;
            } else if (take('%')) {
                const double b = unary();
                if (b == 0) throw std::runtime_error("zero");
                v = std::fmod(v, b);
            } else break;
            if (!std::isfinite(v) || std::abs(v) > 1e100) throw std::runtime_error("range");
        }
        return v;
    }
    double expression() {
        double v = term();
        for (;;) {
            if (take('+')) v += term();
            else if (take('-')) v -= term();
            else break;
            if (!std::isfinite(v) || std::abs(v) > 1e100) throw std::runtime_error("range");
        }
        return v;
    }
public:
    explicit ArithmeticParser(const std::string &text) : s(text) {}
    double run() { const double value = expression(); spaces(); if (p != s.size()) throw std::runtime_error("tail"); return value; }
};

static bool extractCalculation(std::string q, std::string &candidate, std::string &problem) {
    q = trim(q);
    for (const std::string prefix : {"сколько будет", "сколько получится", "посчитай", "вычисли", "рассчитай", "calculate"}) {
        if (q.rfind(prefix, 0) == 0) { q = trim(q.substr(prefix.size())); break; }
    }
    while (!q.empty() && (q.back() == '?' || q.back() == '!' || q.back() == '.' || q.back() == ' ')) q.pop_back();
    bool hasDigit = false;
    for (unsigned char ch : q) {
        if (ch >= '0' && ch <= '9') { hasDigit = true; continue; }
        if (std::string(" \t.()+-*/%^").find(static_cast<char>(ch)) != std::string::npos) continue;
        return false;
    }
    if (!hasDigit || q.find_first_of("+-*/%^()") == std::string::npos) return false;
    try {
        const double value = ArithmeticParser(q).run();
        std::ostringstream out;
        out << std::setprecision(12) << value;
        candidate = out.str();
        problem.clear();
        return true;
    } catch (const std::exception &e) {
        problem = e.what();
        candidate.clear();
        return true;
    }
}

static std::string classify(const std::string &q) {
    if (containsAny(q, {"посчитай", "вычисли", "сколько будет", "сколько получится", "calculate"}) ||
        (q.find_first_of("0123456789") != std::string::npos &&
         q.find_first_of("+-*/%^") != std::string::npos &&
         !containsAny(q, {"версия", "код", "ошибка", "номер"}))) return "calculation";
    if (containsAny(q, {"ошибка", "не работает", "не запускается", "падает", "traceback", "debug", "исправь"})) return "debugging";
    if (containsAny(q, {"код", "программ", "напиши", "создай", "разработай", "c++", "python", "javascript"})) return "code";
    if (containsAny(q, {"план", "по шагам", "с чего начать", "как создать", "как сделать проект"})) return "planning";
    if (containsAny(q, {"объясни", "что такое", "почему", "как работает", "расскажи", "объяснение", "сравни"})) return "explanation";
    if (containsAny(q, {"привет", "здравствуйте", "добрый день", "как дела", "спасибо", "пока"})) return "conversation";
    return "general";
}

static std::vector<std::string> checksFor(const std::string &intent) {
    if (intent == "calculation") return {
        "parse arithmetic with operator precedence",
        "check division by zero and numeric range",
        "verify the evaluated result before explaining it"
    };
    if (intent == "code") return {
        "identify language and expected interface",
        "check syntax, dependencies, and input validation",
        "consider edge cases and propose tests before execution"
    };
    if (intent == "debugging") return {
        "separate observed error text from hypotheses",
        "ask for a reproducible minimal example when needed",
        "check the first failing line and boundary conditions"
    };
    if (intent == "planning") return {
        "define the desired outcome and constraints",
        "split the work into independently testable stages",
        "identify dependencies, failure cases, and acceptance criteria"
    };
    if (intent == "explanation") return {
        "identify the core concept and prior knowledge",
        "connect the mechanism to a concrete example",
        "check for common misconceptions and state limitations"
    };
    if (intent == "conversation") return {
        "recognize social intent",
        "use recent conversational context if it is relevant"
    };
    return {
        "infer the requested outcome without assuming missing facts",
        "inspect the supplied context and identify unknowns",
        "check alternatives and avoid unsupported claims"
    };
}

static void printArray(const std::vector<std::string> &items) {
    std::cout << "[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << "\"" << jsonEscape(items[i]) << "\"";
    }
    std::cout << "]";
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    const std::string query = trim(argv[2]);
    if (query.empty()) return 3;
    const std::string normalized = lowerUtf8(query);
    const std::string intent = classify(normalized);
    const std::vector<std::string> checks = checksFor(intent);
    std::string candidate, calculationProblem;
    const bool hasCalculation = intent == "calculation" && extractCalculation(normalized, candidate, calculationProblem);

    // Machine-readable planning data only. The C++ layer does not fabricate a user-facing
    // prose answer; the trained FROSS Transformer generates that answer in the service.
    std::cout << "{\"engine\":\"FROSS C++ Planner 1.0\","
              << "\"intent\":\"" << jsonEscape(intent) << "\","
              << "\"checks\":";
    printArray(checks);
    std::cout << ",\"candidateResult\":";
    if (hasCalculation && calculationProblem.empty()) std::cout << "\"" << jsonEscape(candidate) << "\"";
    else std::cout << "null";
    std::cout << ",\"calculationIssue\":";
    if (hasCalculation && !calculationProblem.empty()) std::cout << "\"" << jsonEscape(calculationProblem) << "\"";
    else std::cout << "null";
    std::cout << ",\"queryLength\":" << query.size() << "}";
    return 0;
}
