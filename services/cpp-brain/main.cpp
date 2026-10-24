#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <initializer_list>

static std::string lowerAscii(std::string value) {
    for (char &ch : value) {
        unsigned char u = static_cast<unsigned char>(ch);
        if (u >= 'A' && u <= 'Z') ch = static_cast<char>(u - 'A' + 'a');
    }
    return value;
}
static bool has(const std::string &s, const std::string &part) {
    return s.find(part) != std::string::npos;
}
static bool any(const std::string &s, const std::vector<std::string> &parts) {
    for (const auto &part : parts) if (has(s, part)) return true;
    return false;
}
static std::string trim(std::string s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string lastAssistantMessage(const std::string &context) {
    const std::string marker = "fross: ";
    const size_t startMarker = context.rfind(marker);
    if (startMarker == std::string::npos) return "";
    const size_t start = startMarker + marker.size();
    size_t end = context.find("\nПользователь:", start);
    if (end == std::string::npos) end = context.size();
    return trim(context.substr(start, end - start));
}

class Parser {
    const std::string &s; size_t p = 0;
    void spaces() { while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p; }
    bool take(char c) { spaces(); if (p < s.size() && s[p] == c) { ++p; return true; } return false; }
    double primary() {
        spaces();
        if (take('(')) { double v = expression(); if (!take(')')) throw std::runtime_error("paren"); return v; }
        spaces(); size_t start = p;
        while (p < s.size() && ((s[p] >= '0' && s[p] <= '9') || s[p] == '.')) ++p;
        if (p == start) throw std::runtime_error("number");
        double v = std::strtod(s.substr(start, p - start).c_str(), nullptr);
        if (!std::isfinite(v) || std::abs(v) > 1e100) throw std::runtime_error("range");
        return v;
    }
    double power() { double a = primary(); if (take('^')) { double b = unary(); if (std::abs(b) > 100) throw std::runtime_error("exponent"); double v=std::pow(a,b); if(!std::isfinite(v)||std::abs(v)>1e100)throw std::runtime_error("range"); return v; } return a; }
    double unary() { if (take('+')) return unary(); if (take('-')) return -unary(); return power(); }
    double term() {
        double v = unary();
        for (;;) {
            if (take('*')) v *= unary();
            else if (take('/')) { double b = unary(); if (b == 0) throw std::runtime_error("zero"); v /= b; }
            else if (take('%')) { double b = unary(); if (b == 0) throw std::runtime_error("zero"); v = std::fmod(v, b); }
            else break;
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
    explicit Parser(const std::string &text) : s(text) {}
    double run() { double v = expression(); spaces(); if (p != s.size()) throw std::runtime_error("tail"); return v; }
};

static bool calculate(std::string q, std::string &answer) {
    bool prefixed = false;
    for (const std::string prefix : {"сколько будет", "сколько получится", "посчитай", "вычисли", "реши", "calculate"}) {
        if (q.rfind(prefix, 0) == 0) { q = trim(q.substr(prefix.size())); prefixed = true; break; }
    }
    while (!q.empty() && (q.back() == '?' || q.back() == '!' || q.back() == '.' || q.back() == ' ')) q.pop_back();
    bool digit = false;
    for (unsigned char c : q) {
        if (c >= '0' && c <= '9') { digit = true; continue; }
        if (std::string(" \t.()+-*/%^").find(static_cast<char>(c)) != std::string::npos) continue;
        return false;
    }
    if (!digit || (!prefixed && q.find_first_of("+-*/%^()") == std::string::npos)) return false;
    try {
        double value = Parser(q).run();
        std::ostringstream out; out << std::setprecision(12) << value;
        answer = "Ответ: **" + out.str() + "**\n\nВычислено локальным безопасным парсером с проверкой приоритета операций.";
        return true;
    } catch (const std::exception &e) {
        if (std::string(e.what()) == "zero") { answer = "Делить на ноль нельзя: результат не определён."; return true; }
        return false;
    }
}

static std::string definition(const std::string &q) {
    const std::pair<const char *, const char *> entries[] = {
        {"python", "Python — язык программирования с читаемым синтаксисом, часто используемый для автоматизации, веб-сервисов и анализа данных."},
        {"питон", "Python — язык программирования с читаемым синтаксисом, часто используемый для автоматизации, веб-сервисов и анализа данных."},
        {"c++", "C++ — компилируемый язык с высокой производительностью и контролем ресурсов, используемый в системном ПО и игровых движках."},
        {"javascript", "JavaScript — язык для веб-интерактивности и серверных задач. Он не равен Java."},
        {"html", "HTML описывает структуру веб-страницы; CSS отвечает за оформление, а JavaScript — за интерактивность."},
        {"css", "CSS — язык оформления веб-страниц: цвета, шрифты, отступы, сетки и адаптивность."},
        {"алгоритм", "Алгоритм — конечная последовательность чётких шагов с определёнными входами и ожидаемым результатом."},
        {"переменная", "Переменная — именованное значение, которое программа может читать или менять."},
        {"функция", "Функция — отдельный блок кода для конкретной задачи; она может принимать аргументы и возвращать значение."},
        {"нейросеть", "Нейросеть — вычислительная модель с обучаемыми параметрами. Самописные правила этого движка не являются нейросетью."},
        {"искусственный интеллект", "Искусственный интеллект — широкая область методов для автоматизации задач. Система правил может быть полезной, но не равна обученной языковой модели."},
        {"база данных", "База данных — организованное хранилище информации с операциями чтения и изменения записей."},
        {"api", "API — интерфейс, описывающий, как программы обмениваются запросами и ответами."},
        {"http", "HTTP — протокол веб-обмена; HTTPS добавляет защиту соединения через TLS."},
        {"dns", "DNS сопоставляет доменные имена с сетевой информацией, например IP-адресами."},
        {"firestore", "Cloud Firestore — облачная документная база Firebase; правила доступа должны ограничивать данные владельцем аккаунта."},
        {"cloudflare", "Cloudflare предоставляет CDN, DNS, защитные сервисы и Workers для выполнения JavaScript на серверной стороне."}
    };
    for (const auto &entry : entries) if (has(q, entry.first)) return entry.second;
    return "";
}

static std::string codeReply(const std::string &q, bool &matched) {
    matched = true;
    const std::string fence(3, static_cast<char>(96));
    bool cpp = any(q, {"c++", "cpp", "си плюс плюс"});
    bool py = any(q, {"python", "питон", "пайтон"});
    bool fib = any(q, {"фибоначчи", "fibonacci"});
    bool fact = any(q, {"факториал", "factorial"});
    bool calc = any(q, {"калькулятор", "сложи два числа", "арифметик"});
    std::string code, desc, lang;
    if (cpp) {
        lang = "cpp";
        if (fib) { desc = "Первые 10 чисел Фибоначчи."; code = "#include <iostream>\nint main() {\n    long long a = 0, b = 1;\n    for (int i = 0; i < 10; ++i) {\n        std::cout << a << (i == 9 ? '\\n' : ' ');\n        auto next = a + b; a = b; b = next;\n    }\n    return 0;\n}"; }
        else if (fact) { desc = "Факториал числа от 0 до 20."; code = "#include <iostream>\nint main() {\n    int n; if (!(std::cin >> n) || n < 0 || n > 20) return 1;\n    unsigned long long value = 1;\n    for (int i = 2; i <= n; ++i) value *= static_cast<unsigned long long>(i);\n    std::cout << value << std::endl; return 0;\n}"; }
        else if (calc) { desc = "Калькулятор C++ с проверкой деления на ноль."; code = "#include <iostream>\nint main() {\n    double a, b; char op; if (!(std::cin >> a >> op >> b)) return 1;\n    if (op == '+') std::cout << a+b;\n    else if (op == '-') std::cout << a-b;\n    else if (op == '*') std::cout << a*b;\n    else if (op == '/' && b != 0) std::cout << a/b;\n    else std::cout << \"Ошибка операции\";\n    return 0;\n}"; }
        else { desc = "Минимальная программа на C++."; code = "#include <iostream>\nint main() {\n    std::cout << \"Привет, мир!\\n\";\n    return 0;\n}"; }
    } else if (py || any(q, {"код на", "программа на", "простейший код", "пример кода"})) {
        lang = "python";
        if (fib) { desc = "Последовательность Фибоначчи."; code = "def fibonacci(n):\n    result = []\n    a, b = 0, 1\n    for _ in range(max(0, n)):\n        result.append(a)\n        a, b = b, a + b\n    return result\n\nprint(fibonacci(10))"; }
        else if (fact) { desc = "Факториал с проверкой входных данных."; code = "def factorial(n):\n    if n < 0:\n        raise ValueError('n должно быть неотрицательным')\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\nprint(factorial(5))"; }
        else if (calc) { desc = "Калькулятор с проверкой деления на ноль."; code = "def calculate(a, op, b):\n    if op == '+': return a + b\n    if op == '-': return a - b\n    if op == '*': return a * b\n    if op == '/' and b != 0: return a / b\n    raise ValueError('Ошибка операции или деление на ноль')\n\nprint(calculate(8, '+', 4))"; }
        else { desc = "Минимальная программа на Python."; code = "def main():\n    print('Привет, мир!')\n\nif __name__ == '__main__':\n    main()"; }
    } else { matched = false; return ""; }
    return desc + "\n\n" + fence + lang + "\n" + code + "\n" + fence +
        (cpp ? "\n\nСохрани как main.cpp и собери: g++ main.cpp -std=c++17 -o app." : "");
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    const std::string mode = argv[1];
    const std::string q = lowerAscii(trim(argv[2]));
    const std::string context = argc >= 4 ? lowerAscii(trim(argv[3])) : "";
    if (q.empty()) return 3;
    std::string answer, approach;
    if (!context.empty() && any(q, {"почему?", "почему", "а почему", "подробнее", "расскажи подробнее", "продолжи", "продолжай", "а как", "объясни подробнее", "что это значит", "а если"})) {
        const std::string previous = lastAssistantMessage(context);
        if (any(q, {"подробнее", "расскажи подробнее", "продолжи", "продолжай", "объясни подробнее"})) {
            answer = "Продолжу предыдущую тему. Последний ответ был: " + previous.substr(0, 260) +
                     ". Могу разложить его на шаги, привести пример или объяснить отдельный пункт. Укажи, что полезнее.";
        } else if (any(q, {"почему?", "почему", "а почему"}) && any(previous, {"потому что", "так как", "поскольку", "поэтому", "из-за"})) {
            answer = "В предыдущем ответе причина была указана здесь: " + previous.substr(0, 360) +
                     ". Если ты спрашиваешь о другом тезисе, назови его, чтобы я связал объяснение с нужной частью.";
        } else if (any(q, {"а если"})) {
            answer = "Давай проверим другой вариант. Предыдущий ответ: " + previous.substr(0, 220) +
                     ". Напиши условие, которое хочешь изменить, и я сравню результат.";
        } else {
            answer = "Связываю вопрос с предыдущим ответом: " + previous.substr(0, 300) +
                     ". Уточни, к какому утверждению относится вопрос, и я объясню именно его без догадок.";
        }
        approach = "сопоставить короткое уточнение с последним ответом и явно использовать доступный контекст чата.";
    } else if (any(q, {"привет", "здравствуй", "хай", "hello", "hi"})) {
        answer = "Привет! Я на связи. Что разберём?";
        approach = "распознать приветствие и ответить локально.";
    } else if (any(q, {"спасибо", "благодарю", "спс"})) {
        answer = "Пожалуйста! Продолжим, когда понадобится.";
        approach = "распознать благодарность.";
    } else if (any(q, {"кто создатель", "кто создал фросс", "кто автор fross", "кто тебя создал"})) {
        answer = "Автор проекта FROSS: [@Spamton_BigShot_hahaha](https://t.me/Spamton_BigShot_hahaha).";
        approach = "проверить сохранённые сведения о проекте.";
    } else if (any(q, {"твоя модель", "какая у тебя модель", "ты просто гуглишь"})) {
        answer = "Я — FROSS C++ Reasoning Brain 0.6. Это самописный алгоритмический движок, а не обученная нейросеть. Я использую локальные правила и безопасные алгоритмы.";
        approach = "объяснить устройство движка без преувеличений.";
    } else if (calculate(q, answer)) {
        approach = "разобрать арифметику с учётом скобок и приоритета операций.";
    } else if (any(q, {"создай", "напиши", "сделай", "сгенерируй", "код на", "программа на", "пример кода", "простейший код"}) &&
               any(q, {"код", "программ", "python", "питон", "c++", "фибоначчи", "факториал", "калькулятор"})) {
        bool matched = false; answer = codeReply(q, matched); if (!matched) return 3;
        approach = "определить язык и задачу, выбрать шаблон и проверить типичные граничные условия.";
    } else if (any(q, {"что такое", "объясни", "расскажи про", "определение", "как работает", "что означает"})) {
        answer = definition(q); if (answer.empty()) return 3;
        approach = "сопоставить термин с локальной базой определений.";
    } else if (any(q, {"составь план", "помоги спланировать", "как создать сайт", "как разработать сайт"})) {
        answer = "План:\n1. Определить цель и пользователей.\n2. Выбрать функции первой версии.\n3. Спроектировать экраны и данные.\n4. Собрать небольшой прототип.\n5. Проверить ошибки и мобильный экран.\n6. Опубликовать и улучшать по обратной связи.";
        approach = "разбить задачу на этапы и проверки.";
    } else return 3;

    if (mode == "reasoning") {
        answer = "Разбор задачи:\n1. **Цель:** понять желаемый результат.\n2. **Подход:** " + approach + "\n3. **Проверка:** учесть ограничения и не запускать произвольный код из сообщения.\n\n" + answer;
    }
    std::cout << answer;
    return 0;
}
