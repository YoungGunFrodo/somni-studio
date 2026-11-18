"""Google Home, through Home Assistant.

Google offers no local API for its smart-home devices, so there is no honest way to talk to Google
Home directly from a PC. Two paths work, both through Home Assistant:

  1. Add the devices to Home Assistant with their own integrations (Nest, Chromecast, Hue, TP-Link,
     ...): Somni then sees and controls them like any other device. Best: local, fast, two-way.
  2. Add Home Assistant's "Google Assistant SDK" integration: Somni can then send any command to
     Google Assistant ("turn on the bedroom fan"), for devices that only Google knows.

This card has no settings of its own: its status is read from the Home Assistant connection.
"""
from __future__ import annotations

from .base import Integration, State, Status, ToolSpec

SDK_DOMAIN = "google_assistant_sdk"


class GoogleHome(Integration):
    id = "google_home"
    title = "Google Home"
    category = "Smart home"
    description = "Control devices that are set up in the Google Home app, by sending commands to Google Assistant."
    fields = ()
    derived_from = "home_assistant"
    tools = (ToolSpec("google.command", "Tell Google Assistant", "Send a command to Google Assistant."),)
    examples = ("Ask Google to turn on the bedroom fan.", "Tell Google Home to start the vacuum.")
    setup = (
        "Nest Hubs, Nest speakers and Chromecasts connect directly: see Devices at the top.",
        "For lights and plugs that only work through the Google Home app, connect Home Assistant first.",
        "Best: in Home Assistant, add your devices' own integrations (Settings → Devices & services → Add): "
        "Nest, Chromecast, Hue, TP-Link and more. They then work like any other device here.",
        "For devices only Google knows: add Home Assistant's 'Google Assistant SDK' integration and sign in "
        "with your Google account. This card turns green when Somni can send commands to Google.",
    )

    @staticmethod
    def from_home_assistant(ha: Status) -> Status:
        if ha.state is not State.CONNECTED:
            return Status(State.NOT_CONFIGURED, "connect Home Assistant first")
        if SDK_DOMAIN in (ha.details.get("services") or []):
            return Status(State.CONNECTED, "commands go to Google Assistant through Home Assistant")
        return Status(State.NOT_CONFIGURED, "add the Google Assistant SDK integration in Home Assistant")

    def check(self, config: dict) -> Status:          # never called: status is derived
        return Status(State.NOT_CONFIGURED)
