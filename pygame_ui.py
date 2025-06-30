import threading
import time
import os
import csv
import json
import webbrowser

import pygame

# Use the dummy audio driver if no sound card is available.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import config
from main import run_simulation
import utils


pygame.init()
FONT = pygame.font.Font(None, 24)


class InputBox:
    def __init__(self, x, y, w, h, text=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = text
        self.txt_surface = FONT.render(text, True, (0, 0, 0))
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.active = False
            else:
                self.text += event.unicode
            self.txt_surface = FONT.render(self.text, True, (0, 0, 0))

    def draw(self, screen):
        width = max(200, self.txt_surface.get_width() + 10)
        self.rect.w = width
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        pygame.draw.rect(screen, self.color, self.rect, 2)


class Button:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.text_surf = FONT.render(text, True, (0, 0, 0))
        self.color = pygame.Color('gray')

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        screen.blit(self.text_surf, (self.rect.x + 5, self.rect.y + 5))

    def handle_event(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)


def read_scores(run_no):
    path = os.path.join(config.LOGS_DIRECTORY, "wizard_scores.csv")
    scores = []
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if int(row["run"]) == run_no:
                    scores.append(row)
    return scores


def latest_conversation(run_no):
    logs = []
    if os.path.isdir(config.LOGS_DIRECTORY):
        for f in os.listdir(config.LOGS_DIRECTORY):
            if f.endswith('.json') and f.startswith('Wizard') and f'{run_no}.' in f:
                logs.append(f)
    if not logs:
        return None, None
    latest = max(logs, key=lambda f: os.path.getmtime(os.path.join(config.LOGS_DIRECTORY, f)))
    path = os.path.join(config.LOGS_DIRECTORY, latest)
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    return latest, data


def load_summary(run_no):
    path = os.path.join(config.LOGS_DIRECTORY, f"summary_{run_no}.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    return None


def run_system(pop_size, goal, stop_event, pause_event):
    """Run the simulation and allow pausing or cancellation."""
    run_simulation(
        "Generate population",
        pop_size,
        goal,
        stop_event=stop_event,
        pause_event=pause_event,
    )


def main():
    screen = pygame.display.set_mode((900, 700))
    pygame.display.set_caption("AIgentChat UI")
    clock = pygame.time.Clock()

    pop_box = InputBox(250, 40, 140, 32, str(config.POPULATION_SIZE))
    goal_box = InputBox(250, 90, 400, 32, config.WIZARD_DEFAULT_GOAL)
    start_btn = Button(50, 150, 80, 30, "Start")
    pause_btn = Button(150, 150, 80, 30, "Pause")
    cancel_btn = Button(250, 150, 80, 30, "Cancel")

    run_thread = None
    stop_event = threading.Event()
    pause_event = threading.Event()
    run_no = None
    pop_size = config.POPULATION_SIZE
    summary = None
    transcript = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if run_thread and run_thread.is_alive():
                    stop_event.set()
                    run_thread.join()
                running = False
            pop_box.handle_event(event)
            goal_box.handle_event(event)
            if start_btn.handle_event(event) and not (run_thread and run_thread.is_alive()):
                try:
                    pop_size = int(pop_box.text)
                except ValueError:
                    pop_size = config.POPULATION_SIZE
                goal = goal_box.text or config.WIZARD_DEFAULT_GOAL
                run_no = utils.get_run_number() + 1
                summary = None
                transcript = []
                stop_event.clear()
                pause_event.clear()
                run_thread = threading.Thread(target=run_system, args=(pop_size, goal, stop_event, pause_event))
                run_thread.start()
            if pause_btn.handle_event(event) and run_thread and run_thread.is_alive():
                if pause_event.is_set():
                    pause_event.clear()
                else:
                    pause_event.set()
            if cancel_btn.handle_event(event) and run_thread and run_thread.is_alive():
                stop_event.set()

        screen.fill((255, 255, 255))
        # labels
        screen.blit(FONT.render("Population Size:", True, (0, 0, 0)), (50, 50))
        screen.blit(FONT.render("Wizard Goal:", True, (0, 0, 0)), (50, 100))

        pop_box.draw(screen)
        goal_box.draw(screen)
        start_btn.draw(screen)
        pause_btn.draw(screen)
        cancel_btn.draw(screen)

        # progress and logs
        if run_no:
            scores = read_scores(run_no)
            done = len(scores)
            progress = done / float(pop_size)
            pygame.draw.rect(screen, (200, 200, 200), (50, 200, 300, 20), 1)
            pygame.draw.rect(screen, (0, 128, 0), (50, 200, int(300 * progress), 20))
            screen.blit(FONT.render(f"{done}/{pop_size}", True, (0, 0, 0)), (360, 200))
            log_name, data = latest_conversation(run_no)
            if data:
                transcript = data.get("turns", [])

        y = 240
        for turn in transcript[-5:]:
            text = f"{turn['speaker']}: {turn['text'][:60]}"
            screen.blit(FONT.render(text, True, (0, 0, 0)), (50, y))
            y += 20

        if run_thread and not run_thread.is_alive() and run_no and not summary:
            summary = load_summary(run_no)

        if summary:
            screen.blit(FONT.render("Run complete. Summary loaded.", True, (0, 0, 0)), (50, y))
            y += 30
            for idx, entry in enumerate(summary):
                btn = Button(50, y, 120, 25, f"Open {idx+1}")
                btn.draw(screen)
                if idx < len(summary):
                    screen.blit(FONT.render(entry.get('pop_agent_id', ''), True, (0, 0, 0)), (180, y+5))
                if any(ev.type == pygame.MOUSEBUTTONDOWN and btn.rect.collidepoint(ev.pos) for ev in pygame.event.get([pygame.MOUSEBUTTONDOWN])):
                    logs = [f for f in os.listdir(config.LOGS_DIRECTORY) if entry['pop_agent_id'] in f and f.endswith('.json')]
                    if logs:
                        webbrowser.open('file://' + os.path.abspath(os.path.join(config.LOGS_DIRECTORY, logs[0])))
                y += 30

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
