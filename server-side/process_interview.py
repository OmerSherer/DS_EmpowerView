import threading

from classify import process_video_to_csv  # gesture classifyer
from report import make_report  # report maker
from anomalyDetection import analyzeVideo # annomaly detection

def func(app):
    @app.route('/hello')
    def hello():
        return 'hello'
    

def process_interview(file_path):
    def process_interview_thread(file_path):
        fps = process_video_to_csv(input_file=file_path,
                                   model_path='models/my_model6.h5',
                                   output_file_coords='temp_files/interview_outputs/coords.csv',
                                   output_file_confidence='temp_files/interview_outputs/confidence.csv',
                                   show_cam=False)
        
        make_report('temp_files/interview_outputs/confidence.csv')

        print(analyzeVideo('temp_files/interview_outputs/coords.csv', fps))

    interview_thread = threading.Thread(target=process_interview_thread, args=(file_path,))
    interview_thread.start()
