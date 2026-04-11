from novyx_mcp.server import mcp, startup_health_check


def main():
    startup_health_check()
    mcp.run()


if __name__ == "__main__":
    main()
