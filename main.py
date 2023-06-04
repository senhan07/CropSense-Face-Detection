import os
import sys
import cv2
import multiprocessing
from image_processing import process_image
from user_input import select_option, preview_window, clean_output
import variable
from tqdm import tqdm

input_folder = variable.input_folder
output_res = variable.output_res
error_folder = variable.error_folder
preview_output_res = variable.preview_output_res
preview_debug_max_res = variable.preview_debug_max_res

def process_image_worker(args):
    input_file, \
    error_folder, \
    output_folder, \
    debug_output, \
    output_res, \
    preview_output_res, \
    preview_debug_max_res, \
    show_preview, \
    croptype, \
    top_margin_value, \
    bottom_margin_value \
        = args

    error_count = process_image(input_file, \
                                error_folder, \
                                output_folder, \
                                debug_output, \
                                output_res, \
                                preview_output_res, \
                                preview_debug_max_res, \
                                show_preview, \
                                croptype, \
                                top_margin_value, \
                                bottom_margin_value)
    return error_count


if __name__ == '__main__':
    def main():
            top_margin_value, \
            bottom_margin_value, \
            debug_output, \
            output_folder, \
            croptype = select_option()

            os.makedirs(input_folder, exist_ok=True)
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(debug_output, exist_ok=True)
            os.makedirs(error_folder, exist_ok=True)

            show_preview, parallel = preview_window()
            if parallel == True:
                multithread(top_margin_value, \
                            bottom_margin_value, \
                            debug_output, \
                            output_folder, \
                            croptype,\
                            show_preview)
                return
            clean_output(output_folder, debug_output, error_folder)

            input_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
            progress_bar = tqdm(total=len(input_files), desc="[SINGLETHREAD] Processing images", dynamic_ncols=True)
            
            error_count = 0
            for image_path in input_files:
                error_occurred = process_image(image_path,
                                error_folder,
                                output_folder,
                                debug_output,   
                                output_res,
                                preview_output_res,
                                preview_debug_max_res,
                                show_preview,
                                croptype,
                                top_margin_value,
                                bottom_margin_value)
                if error_occurred:
                    error_count += 1
                progress_bar.update(1)

            cv2.destroyAllWindows()

            total_images = len(input_files)
            total_output_images = len(os.listdir(output_folder))
            print(f"Total images: {total_images}")
            print(f"Processed images: {total_images - error_count}")
            print(f"Total faces detected: {total_output_images}")
            print(f"Error images: {error_count}")
        
            return parallel

    def multithread(top_margin_value, \
                    bottom_margin_value, \
                    debug_output, \
                    output_folder, \
                    croptype,\
                    show_preview):
        manager = multiprocessing.Manager()
        error = manager.Value("i", 0)  # Shared variable to track error and error count

        if sys.platform == "win32":
            multiprocessing.freeze_support()

            clean_output(output_folder, debug_output, error_folder)

            # Get a list of input files
            input_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if
                        os.path.isfile(os.path.join(input_folder, filename))]
            worker_args = [
                (
                    input_file,
                    error_folder,
                    output_folder,
                    debug_output,
                    output_res,
                    preview_output_res,
                    preview_debug_max_res,
                    show_preview,
                    croptype,
                    top_margin_value,
                    bottom_margin_value,
                )
                for input_file in input_files
            ]

            # Create a multiprocessing Pool
            pool = multiprocessing.Pool()

            # Apply the worker function to each input file using the multiprocessing Pool and tqdm
            with tqdm(total=len(input_files), desc="[MULTITHREAD] Processing images", dynamic_ncols=True) as progress_bar:
                for error_count in pool.imap_unordered(process_image_worker, worker_args):
                    error.value += error_count
                    progress_bar.update(1)

            cv2.destroyAllWindows()

            # Close the multiprocessing Pool
            pool.close()
            pool.join()

            total_images = len(input_files)
            total_output_images = len(os.listdir(output_folder))
            print(f"Total images: {total_images}")
            print(f"Processed images: {total_images - error.value}")
            print(f"Total faces detected: {total_output_images}")
            print(f"Error images: {error.value}")

    main()