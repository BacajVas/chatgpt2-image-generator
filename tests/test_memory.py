import tempfile
import unittest
import os
from pathlib import Path

from PIL import Image

from core import conversation_context, generate_and_remember
from memory import MemoryStore


class FakeImageModel:
    image_model_id = "test-model"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def generate_image(self, prompt, *, steps, seed):
        self.calls.append((prompt, steps, seed))
        if self.fail:
            raise RuntimeError("model unavailable")
        return Image.new("RGB", (16, 16), "navy")


class MemoryTests(unittest.TestCase):
    def test_success_survives_restart_with_identity_chat_and_image(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(folder)
            store.set_name("Вася")
            store.set_notes("Люблю лодки и мягкий свет")
            store.add_exchange("Помни лодку", "Помню твою идею с лодкой.")
            model = FakeImageModel()

            result = generate_and_remember(
                store, model, "лодка на озере", "фотореализм", 1, 42
            )
            self.assertEqual(model.calls, [("лодка на озере, фотореализм", 1, 42)])
            self.assertTrue(Path(result["image_path"]).is_file())

            reopened = MemoryStore(folder)
            self.assertEqual(reopened.profile()["display_name"], "Вася")
            self.assertIn("лодки", reopened.profile()["notes"])
            self.assertEqual(reopened.profile()["preferred_style"], "фотореализм")
            self.assertEqual(len(reopened.messages()), 2)
            self.assertEqual(reopened.generations()[0]["seed"], 42)
            context = conversation_context(reopened, "Что мы делали?")
            self.assertIn("Вася", context[0]["content"])
            self.assertIn("мягкий свет", context[0]["content"])
            self.assertIn("лодка на озере", context[0]["content"])
            self.assertEqual(context[-1]["content"], "Что мы делали?")

    def test_failed_generation_does_not_enter_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(folder)
            model = FakeImageModel(fail=True)
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                generate_and_remember(store, model, "замок", "акварель", 1, 7)
            self.assertEqual(store.generations(), [])
            self.assertEqual(store.profile()["preferred_style"], "")
            self.assertEqual(list((Path(folder) / "images").iterdir()), [])

    def test_clear_chat_keeps_generated_images(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(folder)
            store.add_exchange("Привет", "Привет!")
            generate_and_remember(store, FakeImageModel(), "дерево", "", 1, 1)
            store.clear_messages()
            self.assertEqual(store.messages(), [])
            self.assertEqual(len(store.generations()), 1)

    def test_old_matching_dialogue_is_recalled(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(folder)
            store.add_exchange("Хочу красную лодку", "Запомнил красную лодку.")
            for number in range(7):
                store.add_exchange(f"Другая тема {number}", "Хорошо")
            context = conversation_context(store, "Помнишь лодку?")
            self.assertIn("красную лодку", context[0]["content"])

    @unittest.skipUnless(os.name == "posix", "POSIX permissions only")
    def test_local_memory_is_private_to_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(Path(folder) / "data")
            self.assertEqual(store.data_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(store.images_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(store.database.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
