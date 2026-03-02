import pygame
import sys

pygame.init()

# Constants
WIDTH, HEIGHT = 360, 540
BLACK = (0, 0, 0)
BG_COLOR = (20, 22, 40)
CARD_COLOR = (33, 35, 60)
WHITE = (240, 240, 240)
LIGHT_PURPLE = (103, 103, 255)
RED = (255, 80, 80)
YELLOW = (255, 220, 90)
GREEN = (100, 255, 150)
GRAY = (180, 180, 200)

FONT = pygame.font.SysFont("segoeui", 16, bold=True)
BIG_FONT = pygame.font.SysFont("segoeui", 18, bold=True)
HEADER_FONT = pygame.font.SysFont("segoeui", 24, bold=True)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
status_messages = []
status_timer = 0
pygame.display.set_caption("AuraSentinel Alerts")

incidents = [
    {"type": "Aggression", "location": "Aisle 7\nBack Left", "risk": "HIGH", "color": RED, "going": 1},
    {"type": "Theft", "location": "Aisle 3\nCenter", "risk": "MEDIUM", "color": YELLOW, "going": 0},
    {"type": "Medical Incident", "location": "Aisle 1\nEntrance", "risk": "HIGH", "color": RED, "going": 0},
    {"type": "Suspicious Behavior", "location": "Aisle 5\nRight Corner", "risk": "LOW", "color": GREEN, "going": 0},
]

current_view = "main"
scroll_offset = 0
selected_index = None
clicked = [False] * len(incidents)
feedback_pending = None
feedback_log = []


def draw_main():
    screen.fill(BG_COLOR)
    header = HEADER_FONT.render("AuraSentinel Alerts", True, WHITE)
    screen.blit(header, (WIDTH // 2 - header.get_width() // 2, 15))

    y = 60 + scroll_offset
    for i, incident in enumerate(incidents):
        mouse_pos = pygame.mouse.get_pos()
        btn_rect = pygame.Rect(30, y + 90, WIDTH - 60, 35)
        hover = btn_rect.collidepoint(mouse_pos)

        card = pygame.Rect(20, y, WIDTH - 40, 130)
        pygame.draw.rect(screen, CARD_COLOR, card, border_radius=10)

        title = BIG_FONT.render(incident['type'], True, WHITE)
        screen.blit(title, (card.x + 10, card.y + 10))

        loc_lines = incident['location'].split('\n')
        for j, line in enumerate(loc_lines):
            loc_text = FONT.render(line, True, GRAY)
            screen.blit(loc_text, (card.x + 10, card.y + 35 + j * 18))

        risk_color = YELLOW if incident['risk'] == "MEDIUM" else (GREEN if incident['risk'] == "LOW" else RED)
        badge = pygame.Rect(card.x + WIDTH - 100, card.y + 10, 70, 22)
        pygame.draw.rect(screen, risk_color, badge, border_radius=6)
        badge_text = FONT.render(incident['risk'], True, BG_COLOR)
        text_x = badge.x + (badge.width - badge_text.get_width()) // 2
        text_y = badge.y + (badge.height - badge_text.get_height()) // 2
        screen.blit(badge_text, (text_x, text_y))

        btn = pygame.Rect(card.x + 10, card.y + 90, card.width - 20, 35)
        btn_color = (130, 130, 255) if hover else LIGHT_PURPLE
        pygame.draw.rect(screen, btn_color, btn, border_radius=6)
        btn_text = FONT.render(
            f"Respond ({incident['going']} going)" if incident['going'] else "Respond", True, WHITE)
        screen.blit(btn_text, (btn.x + (btn.width - btn_text.get_width()) // 2, btn.y + (btn.height - btn_text.get_height()) // 2))

        y += 150

    if not incidents:
        bg_font = pygame.font.SysFont("segoeui", 28, bold=True)
        msg = bg_font.render("Everything is fine!", True, GRAY)
        msg.set_alpha(100)
        screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - msg.get_height() // 2))

    y_offset = HEIGHT - 35
    for status_message in reversed(status_messages[-3:]):
        msg = FONT.render(status_message, True, GREEN)
        msg_surface = pygame.Surface((WIDTH - 40, 25))
        msg_surface.set_alpha(200)
        msg_surface.fill(BG_COLOR)
        screen.blit(msg_surface, (20, y_offset))
        screen.blit(msg, (30, y_offset + 5))
        y_offset -= 30

    pygame.display.flip()


def draw_detail():
    mouse_pos = pygame.mouse.get_pos()
    global btn_return
    screen.fill(BG_COLOR)
    incident = incidents[selected_index]

    header = BIG_FONT.render(f"{incident['type']} at", True, WHITE)
    screen.blit(header, (20, 50))
    location = FONT.render(incident['location'].replace("\n", ", "), True, GRAY)
    screen.blit(location, (20, 80))

    btn_yes = pygame.Rect(30, 150, WIDTH - 60, 55)
    btn_susp = pygame.Rect(30, 220, WIDTH - 60, 55)
    btn_no = pygame.Rect(30, 290, WIDTH - 60, 55)
    global btn_no_area
    btn_no_area = btn_no.copy()

    pygame.draw.rect(screen, (140, 255, 180) if btn_yes.collidepoint(mouse_pos) else GREEN, btn_yes, border_radius=8)
    pygame.draw.rect(screen, (255, 240, 130) if btn_susp.collidepoint(mouse_pos) else YELLOW, btn_susp, border_radius=8)
    pygame.draw.rect(screen, (255, 120, 120) if btn_no.collidepoint(mouse_pos) else RED, btn_no, border_radius=8)
    text_yes = FONT.render("Resolve: Yes", True, BLACK)
    screen.blit(text_yes, (btn_yes.x + (btn_yes.width - text_yes.get_width()) // 2, btn_yes.y + (btn_yes.height - text_yes.get_height()) // 2))
    text_susp = FONT.render("Increased Vigilance Needed", True, BLACK)
    screen.blit(text_susp, (btn_susp.x + (btn_susp.width - text_susp.get_width()) // 2, btn_susp.y + (btn_susp.height - text_susp.get_height()) // 2))
    text_no = FONT.render("No, there's a problem", True, BLACK)
    screen.blit(text_no, (btn_no.x + (btn_no.width - text_no.get_width()) // 2, btn_no.y + (btn_no.height - text_no.get_height()) // 2))

    btn_return = pygame.Rect(10, 10, 80, 30)
    pygame.draw.rect(screen, LIGHT_PURPLE, btn_return, border_radius=6)
    text_return = FONT.render("Return", True, WHITE)
    screen.blit(text_return, (btn_return.x + (btn_return.width - text_return.get_width()) // 2, btn_return.y + (btn_return.height - text_return.get_height()) // 2))

    pygame.display.flip()


def draw_emergency():
    mouse_pos = pygame.mouse.get_pos()
    screen.fill(BG_COLOR)
    incident = incidents[selected_index]
    label = BIG_FONT.render(f"{incident['type']} - Assistance", True, WHITE)
    screen.blit(label, (20, 60))

    if incident['type'] in ["Aggression", "Theft"]:
        call_text = "Call Police (911)"
    elif incident['type'] == "Medical Incident":
        call_text = "Call Ambulance (15)"
    else:
        call_text = "Call Help"

    btn_call = pygame.Rect(30, 150, WIDTH - 60, 60)
    btn_cancel = pygame.Rect(30, 230, WIDTH - 60, 50)

    pygame.draw.rect(screen, (255, 120, 120) if btn_call.collidepoint(mouse_pos) else RED, btn_call, border_radius=8)
    pygame.draw.rect(screen, (60, 60, 90) if btn_cancel.collidepoint(mouse_pos) else CARD_COLOR, btn_cancel, border_radius=8)
    text_call = FONT.render(call_text, True, BLACK)
    screen.blit(text_call, (btn_call.x + (btn_call.width - text_call.get_width()) // 2, btn_call.y + (btn_call.height - text_call.get_height()) // 2))
    text_cancel = FONT.render("Cancel", True, WHITE)
    screen.blit(text_cancel, (btn_cancel.x + (btn_cancel.width - text_cancel.get_width()) // 2, btn_cancel.y + (btn_cancel.height - text_cancel.get_height()) // 2))

    btn_return = pygame.Rect(10, 10, 80, 30)
    pygame.draw.rect(screen, LIGHT_PURPLE, btn_return, border_radius=6)
    screen.blit(FONT.render("Return", True, WHITE), (btn_return.x + 10, btn_return.y + 6))

    pygame.display.flip()


def draw_feedback():
    screen.fill(BG_COLOR)
    if not feedback_pending:
        return

    incident = feedback_pending
    header = HEADER_FONT.render("Was the resolution accurate?", True, WHITE)
    screen.blit(header, (WIDTH // 2 - header.get_width() // 2, 80))

    subtext = FONT.render(f"{incident['incident_type']} at {incident['location'].replace(chr(10), ', ')}", True, GRAY)
    screen.blit(subtext, (WIDTH // 2 - subtext.get_width() // 2, 120))

    btn_yes = pygame.Rect(60, 200, WIDTH - 120, 50)
    btn_no = pygame.Rect(60, 270, WIDTH - 120, 50)
    mouse_pos = pygame.mouse.get_pos()

    pygame.draw.rect(screen, GREEN if btn_yes.collidepoint(mouse_pos) else (100, 255, 150), btn_yes, border_radius=8)
    pygame.draw.rect(screen, RED if btn_no.collidepoint(mouse_pos) else (255, 80, 80), btn_no, border_radius=8)

    text_yes = FONT.render("Accurate", True, BLACK)
    text_no = FONT.render("Not Accurate", True, BLACK)
    screen.blit(text_yes, (btn_yes.centerx - text_yes.get_width() // 2, btn_yes.centery - text_yes.get_height() // 2))
    screen.blit(text_no, (btn_no.centerx - text_no.get_width() // 2, btn_no.centery - text_no.get_height() // 2))

    pygame.display.flip()


def check_main_click(pos):
    global current_view, selected_index
    y = 60 + scroll_offset
    for i, incident in enumerate(incidents):
        btn = pygame.Rect(30, y + 90, WIDTH - 60, 25)
        if btn.collidepoint(pos):
            if not clicked[i]:
                incident['going'] += 1
                clicked[i] = True
            selected_index = i
            current_view = "detail"
            break
        y += 150


def check_detail_click(pos):
    global current_view, selected_index, btn_return, feedback_pending
    if selected_index is None or selected_index >= len(incidents):
        return

    btn_yes = pygame.Rect(40, 150, WIDTH - 80, 45)
    btn_susp = pygame.Rect(40, 210, WIDTH - 80, 45)
    btn_no = btn_no_area

    if btn_return.collidepoint(pos):
        current_view = "main"
        return

    if btn_yes.collidepoint(pos):
        feedback_pending = {
            "incident_type": incidents[selected_index]['type'],
            "location": incidents[selected_index]['location'],
            "resolution": "Resolved"
        }
        incidents.pop(selected_index)
        clicked.pop(selected_index)
        selected_index = None
        current_view = "feedback"
    elif btn_susp.collidepoint(pos):
        selected_index = None
        current_view = "main"
    elif btn_no.collidepoint(pos):
        current_view = "emergency"


def check_emergency_click(pos):
    global status_timer, current_view, selected_index, feedback_pending
    if selected_index is None or selected_index >= len(incidents):
        return

    btn_call = pygame.Rect(40, 150, WIDTH - 80, 50)
    btn_cancel = pygame.Rect(40, 220, WIDTH - 80, 45)
    btn_return = pygame.Rect(10, 10, 80, 30)

    if btn_return.collidepoint(pos):
        current_view = "detail"
        return

    if btn_call.collidepoint(pos):
        service = "Police" if incidents[selected_index]['type'] in ["Aggression", "Theft"] else (
            "Ambulance" if incidents[selected_index]['type'] == "Medical Incident" else "Help")
        zone = incidents[selected_index]['location'].replace("\n", ", ")
        status_messages.append(f"{service} on its way to {zone}")
        status_timer = 180
        feedback_pending = {
            "incident_type": incidents[selected_index]['type'],
            "location": incidents[selected_index]['location'],
            "resolution": f"{service} called"
        }
        incidents.pop(selected_index)
        clicked.pop(selected_index)
        selected_index = None
        current_view = "feedback"
    elif btn_cancel.collidepoint(pos):
        current_view = "detail"


def check_feedback_click(pos):
    global current_view, feedback_pending

    btn_yes = pygame.Rect(60, 200, WIDTH - 120, 50)
    btn_no = pygame.Rect(60, 270, WIDTH - 120, 50)

    if btn_yes.collidepoint(pos):
        feedback_log.append({**feedback_pending, "feedback": "accurate"})
        feedback_pending = None
        current_view = "main"
    elif btn_no.collidepoint(pos):
        feedback_log.append({**feedback_pending, "feedback": "not accurate"})
        feedback_pending = None
        current_view = "main"


def main():
    global current_view, selected_index, scroll_offset, status_timer
    clock = pygame.time.Clock()
    scroll_delay_counter = 0
    while True:
        if selected_index is not None and selected_index >= len(incidents):
            selected_index = None
            current_view = "main"

        for event in pygame.event.get():
            if scroll_delay_counter > 0:
                scroll_delay_counter -= 1
            if event.type == pygame.MOUSEWHEEL:
                scroll_offset += event.y * 30
                max_offset = max(0, len(incidents) * 150 - HEIGHT + 80)
                scroll_offset = max(min(scroll_offset, 0), -max_offset)
                scroll_delay_counter = 6
            if event.type == pygame.MOUSEBUTTONDOWN and scroll_delay_counter > 0:
                continue
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if current_view == "main":
                    check_main_click(event.pos)
                elif current_view == "detail":
                    check_detail_click(event.pos)
                elif current_view == "emergency":
                    check_emergency_click(event.pos)
                elif current_view == "feedback":
                    check_feedback_click(event.pos)

        if current_view == "main":
            draw_main()
        elif current_view == "detail" and selected_index is not None:
            draw_detail()
        elif current_view == "emergency" and selected_index is not None:
            draw_emergency()
        elif current_view == "feedback" and feedback_pending:
            draw_feedback()

        if status_timer > 0:
            status_timer -= 1
        clock.tick(30)


if __name__ == "__main__":
    main()
