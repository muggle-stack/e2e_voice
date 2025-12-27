/**
 * @file mcp_calculator_client.cpp
 * @brief MCP Calculator Client for testing
 *
 * This client connects to the MCP calculator server and tests
 * calculator and counter operations.
 */
#include "mcp_sse_client.h"
#include <iostream>
#include <string>
#include <iomanip>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --url <url>      Server URL (default: http://localhost:8888)" << std::endl;
    std::cout << "  --help           Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    std::string server_url = "http://localhost:8888";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--url" && i + 1 < argc) {
            server_url = argv[++i];
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "MCP Calculator Client" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Connecting to: " << server_url << std::endl;
    std::cout << std::endl;

    // Create client
    mcp::sse_client client(server_url);
    client.set_timeout(10);

    try {
        // Initialize connection
        std::cout << "[1] Initializing connection..." << std::endl;
        bool initialized = client.initialize("CalculatorClient", mcp::MCP_VERSION);

        if (!initialized) {
            std::cerr << "Failed to initialize connection to server" << std::endl;
            return 1;
        }
        std::cout << "    Connected successfully!" << std::endl;

        // Ping server
        std::cout << "\n[2] Pinging server..." << std::endl;
        if (!client.ping()) {
            std::cerr << "Failed to ping server" << std::endl;
            return 1;
        }
        std::cout << "    Server is alive!" << std::endl;

        // Get available tools
        std::cout << "\n[3] Getting available tools..." << std::endl;
        auto tools = client.get_tools();
        std::cout << "    Available tools:" << std::endl;
        for (const auto& tool : tools) {
            std::cout << "    - " << tool.name << ": " << tool.description << std::endl;
        }

        // Test calculator operations
        std::cout << "\n[4] Testing calculator operations..." << std::endl;

        // Add
        {
            mcp::json params = {{"operation", "add"}, {"a", 10}, {"b", 5}};
            mcp::json result = client.call_tool("calculator", params);
            std::cout << "    Add: " << result["content"][0]["text"].get<std::string>() << std::endl;
        }

        // Subtract
        {
            mcp::json params = {{"operation", "subtract"}, {"a", 20}, {"b", 8}};
            mcp::json result = client.call_tool("calculator", params);
            std::cout << "    Subtract: " << result["content"][0]["text"].get<std::string>() << std::endl;
        }

        // Multiply
        {
            mcp::json params = {{"operation", "multiply"}, {"a", 6}, {"b", 7}};
            mcp::json result = client.call_tool("calculator", params);
            std::cout << "    Multiply: " << result["content"][0]["text"].get<std::string>() << std::endl;
        }

        // Divide
        {
            mcp::json params = {{"operation", "divide"}, {"a", 100}, {"b", 4}};
            mcp::json result = client.call_tool("calculator", params);
            std::cout << "    Divide: " << result["content"][0]["text"].get<std::string>() << std::endl;
        }

        // Test counter operations
        std::cout << "\n[5] Testing counter operations..." << std::endl;

        // Reset counter
        {
            mcp::json params = {{"action", "reset"}};
            mcp::json result = client.call_tool("counter", params);
            std::cout << "    " << result["content"][0]["text"].get<std::string>() << std::endl;
        }

        // Increment 3 times
        for (int i = 0; i < 3; i++) {
            mcp::json params = {{"action", "increment"}};
            mcp::json result = client.call_tool("counter", params);
            std::cout << "    " << result["content"][0]["text"].get<std::string>() << std::endl;
        }

        // Increment by 5
        {
            mcp::json params = {{"action", "increment"}, {"value", 5}};
            mcp::json result = client.call_tool("counter", params);
            std::cout << "    " << result["content"][0]["text"].get<std::string>() << std::endl;
        }

        // Decrement by 2
        {
            mcp::json params = {{"action", "decrement"}, {"value", 2}};
            mcp::json result = client.call_tool("counter", params);
            std::cout << "    " << result["content"][0]["text"].get<std::string>() << std::endl;
        }

        // Get current value
        {
            mcp::json params = {{"action", "get"}};
            mcp::json result = client.call_tool("counter", params);
            std::cout << "    " << result["content"][0]["text"].get<std::string>() << std::endl;
        }

        // Set to specific value
        {
            mcp::json params = {{"action", "set"}, {"value", 100}};
            mcp::json result = client.call_tool("counter", params);
            std::cout << "    " << result["content"][0]["text"].get<std::string>() << std::endl;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "All tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const mcp::mcp_exception& e) {
        std::cerr << "MCP error: " << e.what() << " (code: " << static_cast<int>(e.code()) << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
