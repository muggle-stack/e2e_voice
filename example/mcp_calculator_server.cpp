/**
 * @file mcp_calculator_server.cpp
 * @brief MCP Calculator Server with counter functionality
 *
 * This example demonstrates a simple MCP server with:
 * - Basic calculator operations (add, subtract, multiply, divide)
 * - Counter with increment/decrement/reset/get operations
 */
#include "mcp_server.h"
#include "mcp_tool.h"
#include "mcp_resource.h"

#include <iostream>
#include <atomic>
#include <string>
#include <sstream>
#include <iomanip>

// Global counter (thread-safe)
std::atomic<int> g_counter{0};

// Calculator tool handler
mcp::json calculator_handler(const mcp::json& params, const std::string& /* session_id */) {
    if (!params.contains("operation")) {
        throw mcp::mcp_exception(mcp::error_code::invalid_params, "Missing 'operation' parameter");
    }

    std::string operation = params["operation"];
    double result = 0.0;
    std::string operation_str;

    if (operation == "add") {
        if (!params.contains("a") || !params.contains("b")) {
            throw mcp::mcp_exception(mcp::error_code::invalid_params, "Missing 'a' or 'b' parameter");
        }
        double a = params["a"].get<double>();
        double b = params["b"].get<double>();
        result = a + b;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << a << " + " << b << " = " << result;
        operation_str = oss.str();
    } else if (operation == "subtract") {
        if (!params.contains("a") || !params.contains("b")) {
            throw mcp::mcp_exception(mcp::error_code::invalid_params, "Missing 'a' or 'b' parameter");
        }
        double a = params["a"].get<double>();
        double b = params["b"].get<double>();
        result = a - b;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << a << " - " << b << " = " << result;
        operation_str = oss.str();
    } else if (operation == "multiply") {
        if (!params.contains("a") || !params.contains("b")) {
            throw mcp::mcp_exception(mcp::error_code::invalid_params, "Missing 'a' or 'b' parameter");
        }
        double a = params["a"].get<double>();
        double b = params["b"].get<double>();
        result = a * b;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << a << " * " << b << " = " << result;
        operation_str = oss.str();
    } else if (operation == "divide") {
        if (!params.contains("a") || !params.contains("b")) {
            throw mcp::mcp_exception(mcp::error_code::invalid_params, "Missing 'a' or 'b' parameter");
        }
        double a = params["a"].get<double>();
        double b = params["b"].get<double>();
        if (b == 0.0) {
            throw mcp::mcp_exception(mcp::error_code::invalid_params, "Division by zero not allowed");
        }
        result = a / b;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << a << " / " << b << " = " << result;
        operation_str = oss.str();
    } else {
        throw mcp::mcp_exception(mcp::error_code::invalid_params, "Unknown operation: " + operation + ". Supported: add, subtract, multiply, divide");
    }

    std::cout << "[Calculator] " << operation_str << std::endl;

    return {
        {
            {"type", "text"},
            {"text", operation_str}
        }
    };
}

// Counter tool handler
mcp::json counter_handler(const mcp::json& params, const std::string& /* session_id */) {
    if (!params.contains("action")) {
        throw mcp::mcp_exception(mcp::error_code::invalid_params, "Missing 'action' parameter");
    }

    std::string action = params["action"];
    int value = 1;
    if (params.contains("value")) {
        value = params["value"].get<int>();
    }

    int new_value = 0;
    std::string result_str;

    if (action == "increment" || action == "add") {
        new_value = g_counter.fetch_add(value) + value;
        result_str = "Counter incremented by " + std::to_string(value) + ", new value: " + std::to_string(new_value);
    } else if (action == "decrement" || action == "subtract") {
        new_value = g_counter.fetch_sub(value) - value;
        result_str = "Counter decremented by " + std::to_string(value) + ", new value: " + std::to_string(new_value);
    } else if (action == "reset") {
        int reset_to = params.contains("value") ? value : 0;
        g_counter.store(reset_to);
        new_value = reset_to;
        result_str = "Counter reset to " + std::to_string(new_value);
    } else if (action == "get") {
        new_value = g_counter.load();
        result_str = "Current counter value: " + std::to_string(new_value);
    } else if (action == "set") {
        g_counter.store(value);
        new_value = value;
        result_str = "Counter set to " + std::to_string(new_value);
    } else {
        throw mcp::mcp_exception(mcp::error_code::invalid_params,
            "Unknown action: " + action + ". Supported: increment, decrement, reset, get, set");
    }

    std::cout << "[Counter] " << result_str << std::endl;

    return {
        {
            {"type", "text"},
            {"text", result_str}
        }
    };
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --host <host>    Host to bind to (default: localhost)" << std::endl;
    std::cout << "  --port <port>    Port to listen on (default: 8888)" << std::endl;
    std::cout << "  --help           Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    std::string host = "localhost";
    int port = 8888;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        }
    }

    // Create and configure server
    mcp::server::configuration srv_conf;
    srv_conf.host = host;
    srv_conf.port = port;

    mcp::server server(srv_conf);
    server.set_server_info("CalculatorServer", "1.0.0");

    // Set server capabilities
    mcp::json capabilities = {
        {"tools", mcp::json::object()}
    };
    server.set_capabilities(capabilities);

    // Register calculator tool
    mcp::tool calc_tool = mcp::tool_builder("calculator")
        .with_description("Perform basic calculations: add, subtract, multiply, divide")
        .with_string_param("operation", "Operation to perform (add, subtract, multiply, divide)", true)
        .with_number_param("a", "First operand", true)
        .with_number_param("b", "Second operand", true)
        .build();

    // Register counter tool
    mcp::tool counter_tool = mcp::tool_builder("counter")
        .with_description("Manage a counter: increment, decrement, reset, get, set")
        .with_string_param("action", "Action to perform (increment, decrement, reset, get, set)", true)
        .with_number_param("value", "Value for increment/decrement/set (default: 1)", false)
        .build();

    server.register_tool(calc_tool, calculator_handler);
    server.register_tool(counter_tool, counter_handler);

    // Start server
    std::cout << "========================================" << std::endl;
    std::cout << "MCP Calculator Server" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Starting at http://" << host << ":" << port << std::endl;
    std::cout << std::endl;
    std::cout << "Available tools:" << std::endl;
    std::cout << "  - calculator: Basic math operations (add, subtract, multiply, divide)" << std::endl;
    std::cout << "  - counter: Counter management (increment, decrement, reset, get, set)" << std::endl;
    std::cout << std::endl;
    std::cout << "Press Ctrl+C to stop the server" << std::endl;
    std::cout << "========================================" << std::endl;

    server.start(true);  // Blocking mode

    return 0;
}
