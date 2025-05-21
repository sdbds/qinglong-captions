import unittest
from unittest.mock import patch, MagicMock, call
import time

# Assuming the module can be imported like this. Adjust if necessary.
from module.api_handler import (_handle_api_call_with_retries, 
                                _collect_streaming_response, 
                                _post_process_srt_response,
                                extract_code_block_content) # Added for patching
# Add other necessary imports from api_handler if they are used by the helpers directly (e.g. Text for console)
from rich.text import Text
from rich.progress import Progress 
from rich.console import Console # For spec if needed, but MagicMock is usually fine

class TestApiHandlerHelpers(unittest.TestCase):

    @patch('module.api_handler.time.sleep')
    def test_handle_api_call_success_first_try(self, mock_sleep):
        mock_api_func = MagicMock(return_value="success")
        mock_console = MagicMock()
        max_retries = 3
        wait_time = 1
        
        result = _handle_api_call_with_retries(
            mock_api_func, 
            max_retries, 
            wait_time, 
            mock_console,
            api_name="TestAPI_SuccessFirst"
        )
        
        self.assertEqual(result, "success")
        mock_api_func.assert_called_once()
        mock_sleep.assert_not_called()
        # Assert that console.print was not called for errors or retries
        for print_call in mock_console.print.call_args_list:
            args, _ = print_call
            self.assertNotIn("Error processing TestAPI_SuccessFirst", args[0])
            self.assertNotIn("Retrying TestAPI_SuccessFirst", args[0])

    @patch('module.api_handler.time.sleep')
    def test_handle_api_call_success_after_retries(self, mock_sleep):
        mock_api_func = MagicMock()
        mock_console = MagicMock()
        # Simulate failure twice, then success
        mock_api_func.side_effect = [Exception("Attempt 1 Failed"), Exception("Attempt 2 Failed"), "success"]
        max_retries = 3
        wait_time = 2 # Use a distinct wait_time for assertion
        api_name = "TestAPI_RetrySuccess"

        result = _handle_api_call_with_retries(
            mock_api_func, 
            max_retries, 
            wait_time, 
            mock_console,
            api_name=api_name
        )

        self.assertEqual(result, "success")
        self.assertEqual(mock_api_func.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_any_call(wait_time) # Check if sleep was called with the base wait_time

        # Check console messages
        # Error messages
        expected_error_msg1 = f"[red]Error processing {api_name}: {Text('Attempt 1 Failed', style='red')}[/red]"
        expected_error_msg2 = f"[red]Error processing {api_name}: {Text('Attempt 2 Failed', style='red')}[/red]"
        # Retry messages - actual_wait_time could be slightly less than wait_time due to time.time() subtractions
        # So we check for the start of the retry message.
        expected_retry_msg_part = f"[yellow]Retrying {api_name} in "
        
        print_calls = [args[0] for args, _ in mock_console.print.call_args_list]
        self.assertIn(expected_error_msg1, print_calls)
        self.assertIn(expected_error_msg2, print_calls)
        self.assertTrue(any(c.startswith(expected_retry_msg_part) for c in print_calls))


    @patch('module.api_handler.time.sleep')
    def test_handle_api_call_failure_after_all_retries(self, mock_sleep):
        mock_api_func = MagicMock(side_effect=Exception("Persistent Failure"))
        mock_console = MagicMock()
        max_retries = 3
        wait_time = 1
        api_name = "TestAPI_AllFail"

        result = _handle_api_call_with_retries(
            mock_api_func, 
            max_retries, 
            wait_time, 
            mock_console,
            api_name=api_name
        )

        self.assertEqual(result, "") # Designated failure value
        self.assertEqual(mock_api_func.call_count, max_retries)
        self.assertEqual(mock_sleep.call_count, max_retries - 1)

        # Check final failure message
        expected_final_failure_msg = f"[red]Failed to process {api_name} after {max_retries} attempts. Skipping.[/red]"
        
        # Get all arguments passed to console.print
        print_args = [args[0] for args, kwargs in mock_console.print.call_args_list]
        self.assertIn(expected_final_failure_msg, print_args)


    @patch('module.api_handler.time.sleep')
    def test_handle_api_call_429_error_handling(self, mock_sleep):
        mock_api_func = MagicMock()
        mock_console = MagicMock()
        # First call raises 429, second succeeds
        mock_api_func.side_effect = [Exception("Error 429: Too Many Requests"), "success"]
        max_retries = 3
        wait_time = 1 # Initial wait time
        api_name = "TestAPI_429"

        result = _handle_api_call_with_retries(
            mock_api_func, 
            max_retries, 
            wait_time, 
            mock_console,
            api_name=api_name
        )

        self.assertEqual(result, "success")
        self.assertEqual(mock_api_func.call_count, 2)
        mock_sleep.assert_called_once_with(60) # Specific longer wait time for 429

        # Check console message for 429
        expected_429_msg = f"[yellow]Rate limit error for {api_name}. Waiting 60 seconds before retrying...[/yellow]"
        print_calls = [args[0] for args, _ in mock_console.print.call_args_list]
        self.assertIn(expected_429_msg, print_calls)

    @patch('module.api_handler.time.sleep')
    def test_progress_bar_update_on_success(self, mock_sleep):
        mock_api_func = MagicMock(return_value="success")
        mock_console = MagicMock()
        mock_progress = MagicMock(spec=Progress) # Use spec for more accurate mocking
        task_id = "test_task_123"
        on_success_desc = "API call succeeded!"
        
        result = _handle_api_call_with_retries(
            mock_api_func, 
            max_retries=3, 
            wait_time_seconds=1, 
            console=mock_console,
            api_name="TestAPI_Progress",
            progress=mock_progress,
            task_id=task_id,
            on_success_description=on_success_desc
        )
        
        self.assertEqual(result, "success")
        mock_api_func.assert_called_once()
        mock_sleep.assert_not_called()
        mock_progress.update.assert_called_once_with(task_id, description=on_success_desc)

    # --- Tests for _collect_streaming_response ---

    def test_collect_empty_stream(self):
        mock_stream = []
        mock_callback = MagicMock()
        mock_console = MagicMock()

        result = _collect_streaming_response(mock_stream, mock_callback, mock_console)
        
        self.assertEqual(result, "")
        mock_callback.assert_not_called()
        # No progress dots should be printed
        mock_console.print.assert_called_once_with("\n") # Only the final newline

    def test_collect_stream_list_accumulation(self):
        mock_stream_data = ["chunk1_data", "chunk2_data", "chunk3_data"]
        # Simulate callback returning the chunk itself if it's a string
        mock_callback = MagicMock(side_effect=lambda x: x if isinstance(x, str) else None)
        mock_console = MagicMock()

        result = _collect_streaming_response(
            mock_stream_data, 
            mock_callback, 
            mock_console,
            accumulate_as_string=False
        )

        self.assertEqual(result, "chunk1_datachunk2_datachunk3_data")
        self.assertEqual(mock_callback.call_count, len(mock_stream_data))
        mock_callback.assert_any_call("chunk1_data")
        mock_callback.assert_any_call("chunk2_data")
        mock_callback.assert_any_call("chunk3_data")
        
        # Check for progress dots
        expected_dot_calls = [call(".", end="", style="blue")] * len(mock_stream_data)
        expected_dot_calls.append(call("\n")) # Final newline
        mock_console.print.assert_has_calls(expected_dot_calls, any_order=False)

    def test_collect_stream_string_accumulation(self):
        mock_stream_data = ["part1", "part2", "part3"]
        mock_callback = MagicMock(side_effect=lambda x: x)
        mock_console = MagicMock()

        result = _collect_streaming_response(
            mock_stream_data, 
            mock_callback, 
            mock_console,
            accumulate_as_string=True
        )

        self.assertEqual(result, "part1part2part3")
        self.assertEqual(mock_callback.call_count, len(mock_stream_data))
        
        expected_dot_calls = [call(".", end="", style="blue")] * len(mock_stream_data)
        expected_dot_calls.append(call("\n"))
        mock_console.print.assert_has_calls(expected_dot_calls, any_order=False)

    def test_collect_stream_with_none_empty_content(self):
        mock_stream_data = ["valid1", MagicMock(), "valid2", MagicMock()] # Mocks for non-string chunks
        # Callback returns None for non-string, identity for string
        mock_callback = MagicMock(side_effect=lambda x: x if isinstance(x, str) else (None if x == mock_stream_data[1] else ""))
        
        mock_console = MagicMock()

        result = _collect_streaming_response(
            mock_stream_data, 
            mock_callback, 
            mock_console,
            accumulate_as_string=False # Default
        )

        self.assertEqual(result, "valid1valid2")
        self.assertEqual(mock_callback.call_count, len(mock_stream_data))
        # Check console prints: 2 dots for valid content, 1 final newline
        print_calls = mock_console.print.call_args_list
        dot_calls = [c for c in print_calls if c == call(".", end="", style="blue")]
        newline_calls = [c for c in print_calls if c == call("\n")]
        self.assertEqual(len(dot_calls), 2) # Only for "valid1" and "valid2"
        self.assertEqual(len(newline_calls), 1)


    def test_collect_stream_progress_bar_update(self):
        mock_stream_data = ["data"]
        mock_callback = MagicMock(return_value="data_processed")
        mock_console = MagicMock()
        mock_progress = MagicMock(spec=Progress)
        task_id = "task_collect_1"
        desc = "Collecting..."

        _collect_streaming_response(
            mock_stream_data, 
            mock_callback, 
            mock_console, 
            progress=mock_progress, 
            task_id=task_id,
            on_chunk_arrival_description=desc
        )
        
        mock_progress.update.assert_called_once_with(task_id, description=desc)

    # --- Tests for _post_process_srt_response ---

    @patch('module.api_handler.extract_code_block_content')
    def test_post_process_srt_success(self, mock_extract_code_block):
        response_text = "Some text [green]highlighted[/green] ```srt\nSRT CONTENT\n```"
        expected_processed_text = "Some text <font color='green'>highlighted</font> ```srt\nSRT CONTENT\n```"
        expected_srt_content = "SRT CONTENT"
        mock_extract_code_block.return_value = expected_srt_content
        mock_console = MagicMock()

        result = _post_process_srt_response(response_text, mock_console)

        self.assertEqual(result, expected_srt_content)
        mock_extract_code_block.assert_called_once_with(expected_processed_text, "srt", mock_console)

    @patch('module.api_handler.extract_code_block_content')
    def test_post_process_no_srt_found(self, mock_extract_code_block):
        response_text = "Some text without srt block [green]color[/green]"
        expected_processed_text = "Some text without srt block <font color='green'>color</font>"
        mock_extract_code_block.return_value = "" # Simulate no SRT found
        mock_console = MagicMock()

        result = _post_process_srt_response(response_text, mock_console)

        self.assertEqual(result, "")
        mock_extract_code_block.assert_called_once_with(expected_processed_text, "srt", mock_console)
        mock_console.print.assert_called_with("[yellow]Failed to extract SRT content from response.[/yellow]")
        
    @patch('module.api_handler.extract_code_block_content')
    def test_post_process_color_tag_replacement(self, mock_extract_code_block):
        response_text = "Before [green]green text[/green] After. ```srt\n...```"
        expected_text_arg_for_extract = "Before <font color='green'>green text</font> After. ```srt\n...```"
        mock_extract_code_block.return_value = "some srt" # Return value not focus
        mock_console = MagicMock()

        _post_process_srt_response(response_text, mock_console)

        # Check the first argument of the first call to mock_extract_code_block
        self.assertEqual(mock_extract_code_block.call_args[0][0], expected_text_arg_for_extract)

    @patch('module.api_handler.extract_code_block_content')
    def test_post_process_progress_update(self, mock_extract_code_block):
        response_text = "```srt\nCONTENT\n```"
        mock_extract_code_block.return_value = "CONTENT"
        mock_console = MagicMock()
        mock_progress = MagicMock(spec=Progress)
        task_id = "task_post_1"
        desc = "Finalizing media..."

        _post_process_srt_response(
            response_text, 
            mock_console, 
            progress_instance=mock_progress, 
            task_id=task_id,
            final_progress_description=desc
        )
        
        mock_progress.update.assert_called_once_with(task_id, description=desc)

    @patch('module.api_handler.extract_code_block_content')
    def test_post_process_empty_input_response(self, mock_extract_code_block):
        mock_console = MagicMock()
        result = _post_process_srt_response("", mock_console)
        self.assertEqual(result, "")
        mock_extract_code_block.assert_not_called()


if __name__ == '__main__':
    unittest.main()
