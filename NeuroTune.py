# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import librosa
import sounddevice as sd
from pydub import AudioSegment
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTextEdit, QProgressBar, QMessageBox, QFrame,
                             QSpinBox, QComboBox, QTabWidget, QSizePolicy,
                             QCheckBox)
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QFontMetrics
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect

# Localization Strings
TRANSLATION = {
    "ar": {
        "title": "NeuroTune - أداة تحليل وتوليد الصوت",
        "about_tab": "عن التطبيق",
        "file_analysis_tab": "تحليل الملفات",
        "binaural_tab": "مولّد النغمات",
        "realtime_tab": "تحليل الميكروفون",
        "select_file_title": "اختر ملف صوتي",
        "select_file_btn": "اختيار ملف",
        "analyze_btn": "تحليل",
        "file_label": "لم يتم اختيار ملف",
        "processing": "جاري المعالجة:",
        "completed": "اكتمل التحليل:",
        "timeline_label": "المخطط الزمني للتحليل",
        "report_label": "تقرير التحليل",
        "error_analysis": "خطأ في التحليل",
        "error_message": "حدث خطأ أثناء معالجة الملف:",
        "error_retry": "حدث خطأ. الرجاء المحاولة مرة أخرى.",
        "left_channel": "القناة اليسرى",
        "right_channel": "القناة اليمنى",
        "binaural_label": "مولّد النغمات ثنائية الأذن",
        "select_mood": "اختر الحالة النفسية:",
        "generate_button": "تشغيل",
        "stop_button": "إيقاف",
        "realtime_label": "تحليل الميكروفون المباشر",
        "start_mic_analysis": "بدء تحليل الميكروفون",
        "stop_mic_analysis": "إيقاف تحليل الميكروفون",
        "exec_report": "التقرير التنفيذي لـ NeuroTune",
        "audio_file": "ملف الصوت:",
        "duration": "المدة الكلية:",
        "sec": "ثانية",
        "min": "دقيقة",
        "no_effects": "لم يتم العثور على أي مؤثرات عاطفية محددة في هذا المقطع.",
        "left_channel_report": "تحليل القناة اليسرى",
        "right_channel_report": "تحليل القناة اليمنى",
        "unidentified": "غير محدد",
        "calmness": "الهدوء/الاسترخاء",
        "focus": "التركيز/الانتباه",
        "happiness": "السعادة/الحماس",
        "anxiety": "القلق/التوتر",
        "healing": "الشفاء",
        "silence": "فترة صمت أو راحة",
        "uncomfortable_frequencies": "ترددات غير مريحة",
        "dangerous_frequencies": "ترددات خطرة (تحت صوتية)",
        "segment_report_title": "تحليل مقطع صوتي",
        "time_range": "المدة:",
        "from": "من",
        "to": "إلى",
        "mood_detected": "الحالة النفسية:",
        "wave_type": "نوع الموجة:",
        "physical_effect": "التأثير الجسدي:",
        "mental_effect": "التأثير العقلي:",
        "hormonal_effect": "التأثير الهرموني:",
        "danger_level": "درجة الخطورة:",
        "danger_low": "منخفضة",
        "danger_medium": "متوسطة",
        "danger_high": "مرتفعة",
        "danger_very_high": "خطيرة جداً",
        "color_legend": "فهرس الألوان",
        "color_unidentified": "غير محدد",
        "color_calmness": "الهدوء/الاسترخاء",
        "color_focus": "التركيز/الانتباه",
        "color_happiness": "السعادة/الحماس",
        "color_anxiety": "القلق/التوتر",
        "color_healing": "الشفاء",
        "color_silence": "فترة صمت أو راحة",
        "color_uncomfortable": "ترددات غير مريحة",
        "color_dangerous": "ترددات خطرة",
        "channel_select_title": "اختر القنوات للتحليل:",
        "no_channels_selected_error": "الرجاء اختيار قناة واحدة على الأقل للتحليل.",
        "about_content_title": "ما هو NeuroTune؟",
        "about_content_p1": "NeuroTune هي أداة متطورة لتحليل وتوليد الصوت، مصممة لمساعدتك على فهم التأثيرات النفسية والجسدية للأصوات. باستخدام خوارزميات معالجة الإشارات الصوتية المتقدمة، يقوم البرنامج بتحليل الملفات الصوتية أو الصوت المباشر من الميكروفون لتحديد الحالات النفسية المحتملة المرتبطة بها مثل الهدوء، التركيز، أو القلق.",
        "about_content_p2": "البرنامج يعتمد على علم الصوتيات العصبية، حيث يربط بين ترددات الصوت المختلفة وتأثيراتها على موجات الدماغ (مثل موجات ألفا وبيتا) والحالة المزاجية. كما يوفر لك مولدًا للنغمات ثنائية الأذن (Binaural Beats) لإنشاء مؤثرات صوتية مخصصة بناءً على الحالة النفسية التي ترغب في الوصول إليها.",
        "about_content_p3": "تم تطوير هذا البرنامج لتوفير رؤية عميقة حول العلاقة بين الصوت والعقل، وهو مصمم خصيصًا للمهتمين بالصوتيات، العلاج بالموسيقى، والاسترخاء الذهني."
    },
    "en": {
        "title": "NeuroTune - Audio Analysis & Generation Tool",
        "about_tab": "About",
        "file_analysis_tab": "File Analysis",
        "binaural_tab": "Binaural Generator",
        "realtime_tab": "Mic Analysis",
        "select_file_title": "Select Audio File",
        "select_file_btn": "Select File",
        "analyze_btn": "Analyze",
        "file_label": "No file selected",
        "processing": "Processing:",
        "completed": "Analysis completed:",
        "timeline_label": "Analysis Timeline",
        "report_label": "Analysis Report",
        "error_analysis": "Analysis Error",
        "error_message": "An error occurred while processing the file:",
        "error_retry": "An error occurred. Please try again.",
        "left_channel": "Left Channel",
        "right_channel": "Right Channel",
        "binaural_label": "Binaural Beats Generator",
        "select_mood": "Select Mood:",
        "generate_button": "Play",
        "stop_button": "Stop",
        "realtime_label": "Real-time Mic Analysis",
        "start_mic_analysis": "Start Mic Analysis",
        "stop_mic_analysis": "Stop Mic Analysis",
        "exec_report": "NeuroTune Executive Report",
        "audio_file": "Audio File:",
        "duration": "Total Duration:",
        "sec": "sec",
        "min": "min",
        "no_effects": "No specific emotional effects were found in this segment.",
        "left_channel_report": "Left Channel Analysis",
        "right_channel_report": "Right Channel Analysis",
        "unidentified": "unidentified",
        "calmness": "Calmness/Relaxation",
        "focus": "Focus/Attention",
        "happiness": "Happiness/Excitement",
        "anxiety": "Anxiety/Tension",
        "healing": "Healing",
        "silence": "Silence or rest period",
        "uncomfortable_frequencies": "Uncomfortable Frequencies",
        "dangerous_frequencies": "Dangerous (Infrasonic) Frequencies",
        "segment_report_title": "Audio Segment Analysis",
        "time_range": "Duration:",
        "from": "from",
        "to": "to",
        "mood_detected": "Psychological State:",
        "wave_type": "Wave Type:",
        "physical_effect": "Physical Effect:",
        "mental_effect": "Mental Effect:",
        "hormonal_effect": "Hormonal Effect:",
        "danger_level": "Danger Level:",
        "danger_low": "Low",
        "danger_medium": "Medium",
        "danger_high": "High",
        "danger_very_high": "Very High",
        "color_legend": "Color Legend",
        "color_unidentified": "Unidentified",
        "color_calmness": "Calmness/Relaxation",
        "color_focus": "Focus/Attention",
        "color_happiness": "Happiness/Excitement",
        "color_anxiety": "Anxiety/Tension",
        "color_healing": "Healing",
        "color_silence": "Silence or rest period",
        "color_uncomfortable": "Uncomfortable Frequencies",
        "color_dangerous": "Dangerous Frequencies",
        "channel_select_title": "Select Channels for Analysis:",
        "no_channels_selected_error": "Please select at least one channel for analysis.",
        "about_content_title": "What is NeuroTune?",
        "about_content_p1": "NeuroTune is a sophisticated audio analysis and generation tool designed to help you understand the psychological and physiological effects of sound. Using advanced audio signal processing algorithms, the software analyzes audio files or live microphone input to identify potential psychological states associated with them, such as calmness, focus, or anxiety.",
        "about_content_p2": "The program is based on the science of neuro-acoustics, where it links various sound frequencies to their effects on brainwaves (like Alpha and Beta waves) and mood. It also provides a Binaural Beats Generator to create custom audio effects based on the psychological state you wish to achieve.",
        "about_content_p3": "This program was developed to provide deep insight into the relationship between sound and the mind, and is specifically designed for those interested in acoustics, music therapy, and mental relaxation."
    }
}

class TranslationManager:
    def __init__(self, lang="ar"):
        self.lang = lang
    
    def set_language(self, lang):
        self.lang = lang
        
    def get_string(self, key):
        return TRANSLATION.get(self.lang, {}).get(key, key)

# Initialize a global translation manager
TR = TranslationManager()

# Science Database - Enhanced
FREQUENCY_DATA = {
    "calmness": {
        "color": "#2ecc71", "frequency_ranges": [(0.1, 4.0), (4.0, 8.0), (8.0, 12.0), (432.0, 432.0), (528.0, 528.0)],
        "tempo_range": (0, 90), "amplitude_range": (0.0, 0.05), "danger_level": 0,
        "wave_type": {"ar": "موجات دلتا وثيتا وألفا", "en": "Delta, Theta, and Alpha waves"},
        "physical_effect": {"ar": "استرخاء العضلات، تنظيم التنفس، انخفاض معدل ضربات القلب", "en": "Muscle relaxation, regulated breathing, reduced heart rate"},
        "mental_effect": {"ar": "الهدوء، الاسترخاء العميق، التأمل، تقليل الإجهاد", "en": "Calmness, deep relaxation, meditation, stress reduction"},
        "hormonal_effect": {"ar": "زيادة إفراز السيروتونين والأوكسيتوسين", "en": "Increased serotonin and oxytocin secretion"}
    },
    "focus": {
        "color": "#3498db", "frequency_ranges": [(13.0, 15.0), (15.0, 18.0), (30.0, 44.0)],
        "tempo_range": (90, 120), "amplitude_range": (0.05, 0.15), "danger_level": 1,
        "wave_type": {"ar": "موجات بيتا وجاما", "en": "Beta and Gamma waves"},
        "physical_effect": {"ar": "زيادة اليقظة، تحسين التناسق الحركي", "en": "Increased alertness, improved motor coordination"},
        "mental_effect": {"ar": "زيادة التركيز، تحسين الذاكرة، تعزيز القدرة على حل المشكلات", "en": "Increased focus, improved memory, enhanced problem-solving"},
        "hormonal_effect": {"ar": "زيادة إفراز الدوبامين والنورادرينالين", "en": "Increased dopamine and norepinephrine secretion"}
    },
    "happiness": {
        "color": "#f1c40f", "frequency_ranges": [(210.0, 528.0)],
        "tempo_range": (120, 150), "amplitude_range": (0.1, 0.2), "danger_level": 1,
        "wave_type": {"ar": "موجات بيتا", "en": "Beta waves"},
        "physical_effect": {"ar": "زيادة الطاقة، تحفيز الحركة", "en": "Increased energy, stimulated movement"},
        "mental_effect": {"ar": "مشاعر البهجة، الحماس، تحفيز إطلاق الدوبامين", "en": "Feelings of joy, excitement, stimulated dopamine release"},
        "hormonal_effect": {"ar": "زيادة إفراز الدوبامين والإندورفين", "en": "Increased dopamine and endorphin secretion"}
    },
    "anxiety": {
        "color": "#e74c3c", "frequency_ranges": [(18.0, 30.0), (440.0, 528.0)],
        "tempo_range": (130, 200), "amplitude_range": (0.15, 1.0), "danger_level": 2,
        "wave_type": {"ar": "موجات بيتا عالية", "en": "High Beta waves"},
        "physical_effect": {"ar": "زيادة معدل ضربات القلب، توتر العضلات، التعرق", "en": "Increased heart rate, muscle tension, sweating"},
        "mental_effect": {"ar": "القلق، التوتر، الإجهاد، الأرق", "en": "Anxiety, tension, stress, insomnia"},
        "hormonal_effect": {"ar": "زيادة إفراز الكورتيزول والأدرينالين", "en": "Increased cortisol and adrenaline secretion"}
    },
    "healing": {
        "color": "#9b59b6", "frequency_ranges": [(285.0, 285.0), (1111.0, 1111.0)],
        "tempo_range": (60, 100), "amplitude_range": (0.0, 0.1), "danger_level": 0,
        "wave_type": {"ar": "ترددات Solfeggio", "en": "Solfeggio Frequencies"},
        "physical_effect": {"ar": "تجديد الخلايا، تحسين تدفق الدم، تسكين الألم", "en": "Cellular regeneration, improved blood flow, pain relief"},
        "mental_effect": {"ar": "الاسترخاء العميق، الشعور بالسلام الداخلي", "en": "Deep relaxation, a sense of inner peace"},
        "hormonal_effect": {"ar": "توازن إفراز الهرمونات، تقليل الكورتيزول", "en": "Hormonal balance, reduced cortisol"}
    },
    "uncomfortable_frequencies": {
        "color": "#ff7f50", "frequency_ranges": [(20000.0, 22000.0)],
        "tempo_range": (0, 300), "amplitude_range": (0.2, 1.0), "danger_level": 3,
        "wave_type": {"ar": "ترددات عالية جداً", "en": "Very high frequencies"},
        "physical_effect": {"ar": "إجهاد سمعي، صداع، طنين الأذن", "en": "Auditory strain, headache, tinnitus"},
        "mental_effect": {"ar": "انزعاج، توتر، صعوبة في التركيز", "en": "Discomfort, tension, difficulty concentrating"},
        "hormonal_effect": {"ar": "لا يوجد تأثير مباشر وموثق", "en": "No direct documented effect"}
    },
    "dangerous_frequencies": {
        "color": "#c0392b", "frequency_ranges": [(0.0, 20.0)],
        "tempo_range": (0, 60), "amplitude_range": (0.3, 1.0), "danger_level": 5,
        "wave_type": {"ar": "ترددات تحت صوتية", "en": "Infrasonic frequencies"},
        "physical_effect": {"ar": "دوخة، غثيان، ضيق في التنفس", "en": "Dizziness, nausea, shortness of breath"},
        "mental_effect": {"ar": "اضطراب، خوف، شعور بالهلوسة", "en": "Disturbance, fear, feeling of hallucinations"},
        "hormonal_effect": {"ar": "زيادة إفراز الأدرينالين", "en": "Increased adrenaline secretion"}
    },
    "silence": {
        "color": "#6c7a89", "frequency_ranges": [],
        "tempo_range": (0, 0), "amplitude_range": (0.0, 0.01), "danger_level": 0,
        "wave_type": {"ar": "لا يوجد", "en": "None"},
        "physical_effect": {"ar": "استرخاء تام للجسم، هدوء في التنفس", "en": "Complete body relaxation, calm breathing"},
        "mental_effect": {"ar": "سكينة، راحة ذهنية، استعادة النشاط", "en": "Serenity, mental rest, rejuvenation"},
        "hormonal_effect": {"ar": "تحسين توازن الهرمونات", "en": "Improved hormonal balance"}
    },
    "unidentified": {
        "color": "#7f8c8d", "frequency_ranges": [],
        "tempo_range": (0, 0), "amplitude_range": (0.0, 1.0), "danger_level": 0,
        "wave_type": {"ar": "غير معروف", "en": "Unknown"},
        "physical_effect": {"ar": "غير معروف", "en": "Unknown"},
        "mental_effect": {"ar": "غير معروف", "en": "Unknown"},
        "hormonal_effect": {"ar": "غير معروف", "en": "Unknown"}
    }
}

BINAURAL_DATA = {
    "calmness": {"freq1": 432, "freq2": 424, "duration": 30},
    "focus": {"freq1": 130, "freq2": 128, "duration": 30},
    "happiness": {"freq1": 210, "freq2": 204, "duration": 30},
}


class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analysis_results = []
        self.audio_duration = 0.0
        self.setMinimumHeight(250)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.mood_colors = {
            key: data['color'] for key, data in FREQUENCY_DATA.items()
        }

    def set_data(self, results, duration):
        self.analysis_results = results
        self.audio_duration = duration
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1f2937"))
        
        if self.audio_duration == 0 or not self.analysis_results:
            painter.setPen(QColor("#a7f3d0"))
            painter.setFont(QFont("DejaVu Sans Mono", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, TR.get_string("timeline_label"))
            return
            
        width = self.width()
        height = self.height()
        
        channels_to_draw = []
        if any(r['channel'] == 'left' for r in self.analysis_results):
            channels_to_draw.append('left')
        if any(r['channel'] == 'right' for r in self.analysis_results):
            channels_to_draw.append('right')
        if not channels_to_draw and any(r['channel'] == 'mono' for r in self.analysis_results):
            channels_to_draw.append('mono')

        if not channels_to_draw:
            return

        channel_height = height // len(channels_to_draw)
        
        for i, channel_name in enumerate(channels_to_draw):
            channel_rect = QRect(0, i * channel_height, width, channel_height)
            self.draw_channel_timeline(painter, channel_rect, channel_name)
            
            font = QFont("DejaVu Sans Mono", 12, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QColor("#a7f3d0"))
            label_text = TR.get_string("left_channel") if channel_name == 'left' else TR.get_string("right_channel")
            if channel_name == 'mono': label_text = TR.get_string("left_channel")
            painter.drawText(channel_rect.adjusted(10, 10, -10, -10), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, label_text)

    def draw_channel_timeline(self, painter, rect, channel_name):
        width = rect.width()
        y_offset = rect.y()
        height = rect.height()
        
        painter.fillRect(rect, QColor("#111827"))
        
        channel_results = [r for r in self.analysis_results if r['channel'] == channel_name]
        
        if not channel_results:
            return
            
        for result in channel_results:
            start_time = result['start_time']
            end_time = result['end_time']
            mood_key = result['mood']
            
            # Use standardized keys for color lookup
            mood_color = QColor(self.mood_colors.get(mood_key, "#7f8c8d"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(mood_color))
            x1 = int((start_time / self.audio_duration) * width)
            x2 = int((end_time / self.audio_duration) * width)
            segment_width = max(1, x2 - x1)
            painter.drawRect(x1, y_offset, segment_width, height)
            
            font = QFont("DejaVu Sans Mono", 8)
            painter.setFont(font)
            painter.setPen(QColor("#a7f3d0"))
            
            start_text = f"{start_time:.1f}s"
            fm = QFontMetrics(font)
            start_text_width = fm.horizontalAdvance(start_text)
            if x1 + start_text_width < width:
                painter.drawText(x1, y_offset + height - 5, start_text)
            
            end_text = f"{end_time:.1f}s"
            end_text_width = fm.horizontalAdvance(end_text)
            if x2 - end_text_width > 0 and x2 > x1:
                 painter.drawText(x2 - end_text_width, y_offset + height - 5, end_text)

class AnalysisThread(QThread):
    progress_signal = pyqtSignal(int)
    results_signal = pyqtSignal(list, float)
    error_signal = pyqtSignal(str)
    
    def __init__(self, file_path, channels_to_analyze):
        super().__init__()
        self.file_path = file_path
        self.channels_to_analyze = channels_to_analyze
        self.weights = {'freq': 10, 'tempo': 10, 'amp': 10}

    def run(self):
        try:
            audio = AudioSegment.from_file(self.file_path)
            duration_ms = len(audio)
            
            y, sr = librosa.load(self.file_path, sr=None, mono=False)
            
            analyzable_channels = []
            if audio.channels == 2:
                if 'left' in self.channels_to_analyze:
                    analyzable_channels.append(('left', audio.split_to_mono()[0], y[0]))
                if 'right' in self.channels_to_analyze:
                    analyzable_channels.append(('right', audio.split_to_mono()[1], y[1]))
            else:
                if 'left' in self.channels_to_analyze or 'right' in self.channels_to_analyze:
                     analyzable_channels.append(('mono', audio, y))

            analysis_results = []
            segment_duration_sec = 5
            segment_duration_ms = segment_duration_sec * 1000
            num_segments = int(duration_ms / segment_duration_ms)

            for i in range(num_segments):
                self.progress_signal.emit(int((i + 1) / num_segments * 100))
                
                for channel_name, channel_audio, channel_y in analyzable_channels:
                    start_ms = i * segment_duration_ms
                    end_ms = start_ms + segment_duration_ms
                    segment_audio = channel_audio[start_ms:end_ms]
                    samples = np.array(segment_audio.get_array_of_samples())
                    segment_y = channel_y[int(start_ms/1000 * sr) : int(end_ms/1000 * sr)]
                    
                    rms = librosa.feature.rms(y=segment_y)
                    mean_amplitude = np.mean(rms)
                    
                    if mean_amplitude < 0.01:
                        dominant_mood = "silence"
                        dominant_freq = 0
                        tempo = 0
                    else:
                        n = len(samples)
                        freq_domain = np.fft.fft(samples)
                        frequencies = np.fft.fftfreq(n, d=1/segment_audio.frame_rate)
                        dominant_freq_index = np.argmax(np.abs(freq_domain[1:n//2]))
                        dominant_freq = abs(frequencies[dominant_freq_index])
                        tempo = librosa.beat.beat_track(y=segment_y, sr=sr, units='time')[0]
                        
                        mood_scores = {}
                        for mood_key, data in FREQUENCY_DATA.items():
                            if mood_key == "silence" or mood_key == "unidentified":
                                continue
                            score = 0
                            for freq_range in data["frequency_ranges"]:
                                if freq_range[0] <= dominant_freq <= freq_range[1]: score += self.weights['freq']; break
                            if data["tempo_range"][0] <= tempo <= data["tempo_range"][1]: score += self.weights['tempo']
                            if data["amplitude_range"][0] <= mean_amplitude <= data["amplitude_range"][1]: score += self.weights['amp']
                            mood_scores[mood_key] = score
                        
                        if mood_scores and max(mood_scores.values()) > 0:
                            dominant_mood = max(mood_scores, key=mood_scores.get)
                        else:
                            dominant_mood = "unidentified"

                    analysis_results.append({
                        "channel": channel_name,
                        "start_time": start_ms / 1000, "end_time": end_ms / 1000,
                        "mood": dominant_mood
                    })
            
            consolidated_results = self.consolidate_results(analysis_results)
            self.results_signal.emit(consolidated_results, duration_ms / 1000)

        except Exception as e:
            self.error_signal.emit(str(e))
    
    def consolidate_results(self, results):
        if not results:
            return []
            
        consolidated = []
        if not results:
            return consolidated

        left_results = [r for r in results if r['channel'] == 'left']
        right_results = [r for r in results if r['channel'] == 'right']
        mono_results = [r for r in results if r['channel'] == 'mono']
        
        if left_results:
            consolidated.extend(self._consolidate_channel(left_results))
        if right_results:
            consolidated.extend(self._consolidate_channel(right_results))
        if mono_results:
            consolidated.extend(self._consolidate_channel(mono_results))

        return consolidated
        
    def _consolidate_channel(self, results):
        if not results:
            return []
            
        consolidated = []
        current_segment = results[0]
        
        for i in range(1, len(results)):
            next_segment = results[i]
            if next_segment['mood'] == current_segment['mood']:
                current_segment['end_time'] = next_segment['end_time']
            else:
                consolidated.append(current_segment)
                current_segment = next_segment
                
        consolidated.append(current_segment)
        return consolidated


class NeuroTuneApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_path = ""
        self.analysis_results = []
        self.audio_duration = 0.0
        self.realtime_thread = None
        self.binaural_thread = None
        self.initUI()
        self.update_ui_language()

    def initUI(self):
        self.setWindowTitle(TR.get_string("title"))
        self.setMinimumSize(1000, 800)
        
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #1f2937; color: #a7f3d0; font-family: 'DejaVu Sans Mono';")
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        header_frame = QFrame()
        header_frame.setStyleSheet("QFrame { background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 10px; }")
        header_layout = QHBoxLayout()
        header_frame.setLayout(header_layout)

        title_label = QLabel("NeuroTune")
        title_label.setFont(QFont("DejaVu Sans Mono", 30, QFont.Weight.ExtraBold))
        title_label.setStyleSheet("color: #a7f3d0; padding: 10px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()

        self.language_combo = QComboBox()
        self.language_combo.setObjectName("language_combo")
        self.language_combo.addItems(["العربية", "English"])
        self.language_combo.setStyleSheet("""
            QComboBox { background-color: #111827; color: #a7f3d0; border-radius: 5px; padding: 5px; border: 1px solid #374151; }
            QComboBox::drop-down { border: none; }
        """)
        self.language_combo.currentIndexChanged.connect(self.change_language)
        header_layout.addWidget(self.language_combo)
        
        main_layout.addWidget(header_frame)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("tab_widget")
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { background-color: #1f2937; border: none; }
            QTabBar::tab { background: #111827; color: #a7f3d0; padding: 10px; border-top-left-radius: 8px; border-top-right-radius: 8px; border: 1px solid #374151; margin-bottom: -1px;}
            QTabBar::tab:selected { background: #1f2937; border: 1px solid #a7f3d0; border-bottom-color: #1f2937; }
        """)
        main_layout.addWidget(self.tab_widget, 1)
        
        self.about_tab = QWidget()
        self.file_analysis_tab = QWidget()
        self.binaural_generator_tab = QWidget()
        self.realtime_analysis_tab = QWidget()

        self.tab_widget.addTab(self.about_tab, TR.get_string("about_tab"))
        self.tab_widget.addTab(self.file_analysis_tab, TR.get_string("file_analysis_tab"))
        self.tab_widget.addTab(self.binaural_generator_tab, TR.get_string("binaural_tab"))
        self.tab_widget.addTab(self.realtime_analysis_tab, TR.get_string("realtime_tab"))
        
        self.create_about_section(self.about_tab)
        self.create_file_analysis_section(self.file_analysis_tab)
        self.create_binaural_generator_section(self.binaural_generator_tab)
        self.create_realtime_analysis_section(self.realtime_analysis_tab)
        
        self.create_results_section(main_layout)

    def create_about_section(self, parent_widget):
        main_layout = QVBoxLayout()
        parent_widget.setLayout(main_layout)
        about_frame = QFrame()
        about_frame.setStyleSheet("QFrame { background: #111827; border: 1px solid #374151; border-radius: 10px; padding: 15px; } QLabel { color: #a7f3d0; }")
        about_layout = QVBoxLayout()
        about_frame.setLayout(about_layout)
        
        about_title = QLabel(TR.get_string("about_content_title"))
        about_title.setObjectName("about_content_title")
        about_title.setFont(QFont("DejaVu Sans Mono", 16, QFont.Weight.Bold))
        about_layout.addWidget(about_title)
        
        about_text = QTextEdit()
        about_text.setObjectName("about_text")
        about_text.setReadOnly(True)
        about_text.setStyleSheet("QTextEdit { background-color: #1f2937; border: none; padding: 0; }")
        about_text.setFont(QFont("DejaVu Sans Mono", 12))
        about_layout.addWidget(about_text)
        
        main_layout.addWidget(about_frame)
        main_layout.addStretch()

    def create_file_analysis_section(self, parent_widget):
        main_layout = QVBoxLayout()
        parent_widget.setLayout(main_layout)

        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame { background: #111827; border: 1px solid #374151; border-radius: 10px; padding: 15px; }
            QLabel { color: #a7f3d0; }
        """)
        control_layout = QVBoxLayout()
        control_frame.setLayout(control_layout)
        
        title_label = QLabel(TR.get_string("select_file_title"))
        title_label.setObjectName("select_file_title")
        title_label.setFont(QFont("DejaVu Sans Mono", 16, QFont.Weight.Bold))
        control_layout.addWidget(title_label)
        
        file_h_layout = QHBoxLayout()
        self.file_label = QLabel(TR.get_string("file_label"))
        self.file_label.setObjectName("file_label")
        self.file_label.setFont(QFont("DejaVu Sans Mono", 12))
        self.file_label.setStyleSheet("color: #6ee7b7;")
        file_h_layout.addWidget(self.file_label, 1)

        self.btn_open = QPushButton(TR.get_string("select_file_btn"))
        self.btn_open.setObjectName("btn_open")
        self.btn_open.setStyleSheet("QPushButton { background-color: #374151; color: #a7f3d0; border-radius: 5px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #4b5563; } QPushButton:pressed { background-color: #6b7280; }")
        self.btn_open.clicked.connect(self.open_file)
        file_h_layout.addWidget(self.btn_open)

        self.btn_analyze = QPushButton(TR.get_string("analyze_btn"))
        self.btn_analyze.setObjectName("btn_analyze")
        self.btn_analyze.setStyleSheet("QPushButton { background-color: #10b981; color: #000; border-radius: 5px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #059669; } QPushButton:pressed { background-color: #047857; } QPushButton:disabled { background-color: #4b5563; color: #9ca3af; }")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        file_h_layout.addWidget(self.btn_analyze)
        
        control_layout.addLayout(file_h_layout)

        channel_select_layout = QVBoxLayout()
        channel_select_title = QLabel(TR.get_string("channel_select_title"))
        channel_select_title.setObjectName("channel_select_title")
        channel_select_title.setFont(QFont("DejaVu Sans Mono", 12))
        channel_select_layout.addWidget(channel_select_title)

        checkbox_h_layout = QHBoxLayout()
        self.checkbox_left = QCheckBox(TR.get_string("left_channel"))
        self.checkbox_left.setObjectName("checkbox_left")
        self.checkbox_left.setChecked(True)
        self.checkbox_left.setStyleSheet("QCheckBox { color: #a7f3d0; } QCheckBox::indicator { border: 1px solid #6b7280; background-color: #374151; border-radius: 3px; } QCheckBox::indicator:checked { background-color: #6ee7b7; border-color: #6ee7b7; }")
        checkbox_h_layout.addWidget(self.checkbox_left)

        self.checkbox_right = QCheckBox(TR.get_string("right_channel"))
        self.checkbox_right.setObjectName("checkbox_right")
        self.checkbox_right.setChecked(True)
        self.checkbox_right.setStyleSheet("QCheckBox { color: #a7f3d0; } QCheckBox::indicator { border: 1px solid #6b7280; background-color: #374151; border-radius: 3px; } QCheckBox::indicator:checked { background-color: #6ee7b7; border-color: #6ee7b7; }")
        checkbox_h_layout.addWidget(self.checkbox_right)
        checkbox_h_layout.addStretch()

        channel_select_layout.addLayout(checkbox_h_layout)
        control_layout.addLayout(channel_select_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 2px solid #374151; border-radius: 5px; background-color: #111827; text-align: center; color: #a7f3d0; }
            QProgressBar::chunk { background-color: #10b981; border-radius: 3px; }
        """)
        control_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(control_frame)

    def create_realtime_analysis_section(self, parent_widget):
        main_layout = QVBoxLayout()
        parent_widget.setLayout(main_layout)
        
        mic_frame = QFrame()
        mic_frame.setStyleSheet("QFrame { background: #111827; border: 1px solid #374151; border-radius: 10px; padding: 15px; } QLabel { color: #a7f3d0; }")
        mic_layout = QHBoxLayout()
        mic_frame.setLayout(mic_layout)
        
        title_label = QLabel(TR.get_string("realtime_label"))
        title_label.setObjectName("realtime_label_title")
        title_label.setFont(QFont("DejaVu Sans Mono", 16, QFont.Weight.Bold))
        mic_layout.addWidget(title_label, 1)
        
        self.btn_mic_start = QPushButton(TR.get_string("start_mic_analysis"))
        self.btn_mic_start.setObjectName("btn_mic_start")
        self.btn_mic_start.setStyleSheet("QPushButton { background-color: #10b981; color: #000; border-radius: 5px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #059669; }")
        self.btn_mic_start.clicked.connect(self.start_mic_analysis)
        mic_layout.addWidget(self.btn_mic_start)
        
        self.btn_mic_stop = QPushButton(TR.get_string("stop_mic_analysis"))
        self.btn_mic_stop.setObjectName("btn_mic_stop")
        self.btn_mic_stop.setStyleSheet("QPushButton { background-color: #ef4444; color: #fff; border-radius: 5px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #dc2626; }")
        self.btn_mic_stop.clicked.connect(self.stop_mic_analysis)
        self.btn_mic_stop.setEnabled(False)
        mic_layout.addWidget(self.btn_mic_stop)
        
        main_layout.addWidget(mic_frame)
        main_layout.addStretch()

    def create_binaural_generator_section(self, parent_widget):
        main_layout = QVBoxLayout()
        parent_widget.setLayout(main_layout)

        binaural_frame = QFrame()
        binaural_frame.setStyleSheet("QFrame { background: #111827; border: 1px solid #374151; border-radius: 10px; padding: 15px; } QLabel { color: #a7f3d0; }")
        binaural_layout = QHBoxLayout()
        binaural_frame.setLayout(binaural_layout)

        title_label = QLabel(TR.get_string("binaural_label"))
        title_label.setObjectName("binaural_label_title")
        title_label.setFont(QFont("DejaVu Sans Mono", 16, QFont.Weight.Bold))
        binaural_layout.addWidget(title_label, 1)

        mood_label = QLabel(TR.get_string("select_mood"))
        mood_label.setObjectName("select_mood_label")
        binaural_layout.addWidget(mood_label)
        
        self.binaural_mood_combo = QComboBox()
        self.binaural_mood_combo.setObjectName("binaural_mood_combo")
        self.binaural_mood_combo.setStyleSheet("QComboBox { background-color: #374151; color: #a7f3d0; border-radius: 5px; padding: 5px; border: 1px solid #4b5563; }")
        self.update_binaural_mood_combo()
        binaural_layout.addWidget(self.binaural_mood_combo)
        
        self.btn_binaural_start = QPushButton(TR.get_string("generate_button"))
        self.btn_binaural_start.setObjectName("btn_binaural_start")
        self.btn_binaural_start.setStyleSheet("QPushButton { background-color: #10b981; color: #000; border-radius: 5px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #059669; }")
        self.btn_binaural_start.clicked.connect(self.start_binaural_generator)
        binaural_layout.addWidget(self.btn_binaural_start)

        self.btn_binaural_stop = QPushButton(TR.get_string("stop_button"))
        self.btn_binaural_stop.setObjectName("btn_binaural_stop")
        self.btn_binaural_stop.setStyleSheet("QPushButton { background-color: #ef4444; color: #fff; border-radius: 5px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #dc2626; }")
        self.btn_binaural_stop.clicked.connect(self.stop_binaural_generator)
        self.btn_binaural_stop.setEnabled(False)
        binaural_layout.addWidget(self.btn_binaural_stop)
        
        main_layout.addWidget(binaural_frame)
        main_layout.addStretch()

    def create_results_section(self, layout):
        results_frame = QFrame()
        results_frame.setStyleSheet("background-color: #111827; border: 1px solid #374151; border-radius: 10px; padding: 15px;")
        results_layout = QVBoxLayout()
        results_frame.setLayout(results_layout)
        layout.addWidget(results_frame, 2)

        top_results_layout = QHBoxLayout()
        results_layout.addLayout(top_results_layout)

        timeline_widget_container = QWidget()
        timeline_layout = QVBoxLayout()
        timeline_widget_container.setLayout(timeline_layout)
        
        timeline_label = QLabel(TR.get_string("timeline_label"))
        timeline_label.setObjectName("timeline_label_title")
        timeline_label.setFont(QFont("DejaVu Sans Mono", 16, QFont.Weight.Bold))
        timeline_label.setStyleSheet("color: #a7f3d0;")
        timeline_layout.addWidget(timeline_label)

        self.timeline_widget = TimelineWidget()
        self.timeline_widget.setObjectName("timeline_widget")
        self.timeline_widget.setMinimumHeight(250)
        self.timeline_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        timeline_layout.addWidget(self.timeline_widget)
        top_results_layout.addWidget(timeline_widget_container, 3)

        color_legend_frame = QFrame()
        color_legend_frame.setStyleSheet("background-color: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 10px;")
        color_legend_layout = QVBoxLayout()
        color_legend_frame.setLayout(color_legend_layout)
        top_results_layout.addWidget(color_legend_frame, 1)

        legend_title = QLabel(TR.get_string("color_legend"))
        legend_title.setObjectName("color_legend_title")
        legend_title.setFont(QFont("DejaVu Sans Mono", 14, QFont.Weight.Bold))
        legend_title.setStyleSheet("color: #a7f3d0; margin-bottom: 10px;")
        color_legend_layout.addWidget(legend_title)
        
        self.legend_labels = []
        for mood_key, mood_data in FREQUENCY_DATA.items():
            legend_item_layout = QHBoxLayout()
            color_label = QLabel("■")
            color_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            color_label.setStyleSheet(f"color: {mood_data['color']};")
            legend_item_layout.addWidget(color_label)

            text_label = QLabel("")
            text_label.setObjectName(f"legend_label_{mood_key}")
            text_label.setStyleSheet("color: #d1d5db; font-size: 10px;")
            self.legend_labels.append(text_label)
            legend_item_layout.addWidget(text_label)

            legend_item_layout.addStretch()
            color_legend_layout.addLayout(legend_item_layout)

        report_label = QLabel(TR.get_string("report_label"))
        report_label.setObjectName("report_label_title")
        report_label.setFont(QFont("DejaVu Sans Mono", 16, QFont.Weight.Bold))
        report_label.setStyleSheet("color: #a7f3d0; margin-top: 10px;")
        results_layout.addWidget(report_label)

        self.report_text = QTextEdit()
        self.report_text.setObjectName("report_text")
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("DejaVu Sans Mono", 12))
        self.report_text.setStyleSheet("QTextEdit { background-color: #111827; color: #a7f3d0; border: 1px solid #374151; border-radius: 8px; padding: 15px; }")
        results_layout.addWidget(self.report_text, 1)

    def change_language(self, index):
        lang_code = "en" if index == 1 else "ar"
        TR.set_language(lang_code)
        self.update_ui_language()

    def update_ui_language(self):
        self.setWindowTitle(TR.get_string("title"))
        
        self.tab_widget.setTabText(0, TR.get_string("about_tab"))
        self.tab_widget.setTabText(1, TR.get_string("file_analysis_tab"))
        self.tab_widget.setTabText(2, TR.get_string("binaural_tab"))
        self.tab_widget.setTabText(3, TR.get_string("realtime_tab"))

        # Update file analysis tab
        if self.findChild(QLabel, "select_file_title"):
            self.findChild(QLabel, "select_file_title").setText(TR.get_string("select_file_title"))
        if self.findChild(QLabel, "file_label"):
            self.findChild(QLabel, "file_label").setText(TR.get_string("file_label"))
        if self.findChild(QPushButton, "btn_open"):
            self.findChild(QPushButton, "btn_open").setText(TR.get_string("select_file_btn"))
        if self.findChild(QPushButton, "btn_analyze"):
            self.findChild(QPushButton, "btn_analyze").setText(TR.get_string("analyze_btn"))
        if self.findChild(QCheckBox, "checkbox_left"):
            self.findChild(QCheckBox, "checkbox_left").setText(TR.get_string("left_channel"))
        if self.findChild(QCheckBox, "checkbox_right"):
            self.findChild(QCheckBox, "checkbox_right").setText(TR.get_string("right_channel"))
        if self.findChild(QLabel, "channel_select_title"):
            self.findChild(QLabel, "channel_select_title").setText(TR.get_string("channel_select_title"))

        # Update binaural tab
        if self.findChild(QLabel, "binaural_label_title"):
            self.findChild(QLabel, "binaural_label_title").setText(TR.get_string("binaural_label"))
        if self.findChild(QLabel, "select_mood_label"):
            self.findChild(QLabel, "select_mood_label").setText(TR.get_string("select_mood"))
        if self.findChild(QPushButton, "btn_binaural_start"):
            self.findChild(QPushButton, "btn_binaural_start").setText(TR.get_string("generate_button"))
        if self.findChild(QPushButton, "btn_binaural_stop"):
            self.findChild(QPushButton, "btn_binaural_stop").setText(TR.get_string("stop_button"))
        
        self.update_binaural_mood_combo()

        # Update realtime tab
        if self.findChild(QLabel, "realtime_label_title"):
            self.findChild(QLabel, "realtime_label_title").setText(TR.get_string("realtime_label"))
        if self.findChild(QPushButton, "btn_mic_start"):
            self.findChild(QPushButton, "btn_mic_start").setText(TR.get_string("start_mic_analysis"))
        if self.findChild(QPushButton, "btn_mic_stop"):
            self.findChild(QPushButton, "btn_mic_stop").setText(TR.get_string("stop_mic_analysis"))

        # Update results section
        if self.findChild(QLabel, "timeline_label_title"):
            self.findChild(QLabel, "timeline_label_title").setText(TR.get_string("timeline_label"))
        if self.findChild(QLabel, "report_label_title"):
            self.findChild(QLabel, "report_label_title").setText(TR.get_string("report_label"))
        if self.findChild(QLabel, "color_legend_title"):
            self.findChild(QLabel, "color_legend_title").setText(TR.get_string("color_legend"))
        
        # Update legend labels
        for mood_key, mood_data in FREQUENCY_DATA.items():
            label = self.findChild(QLabel, f"legend_label_{mood_key}")
            if label:
                label.setText(TR.get_string(mood_key))

        # Update about text
        about_text = self.findChild(QTextEdit, "about_text")
        if about_text:
            content = f"<h4 style='color:#a7f3d0;'>{TR.get_string('about_content_title')}</h4>"
            content += f"<p style='color:#d1d5db;'>{TR.get_string('about_content_p1')}</p>"
            content += f"<p style='color:#d1d5db;'>{TR.get_string('about_content_p2')}</p>"
            content += f"<p style='color:#d1d5db;'>{TR.get_string('about_content_p3')}</p>"
            about_text.setHtml(content)


        # Re-generate report and timeline to update their content with the new language
        if self.analysis_results:
            self.generate_report()
            self.timeline_widget.set_data(self.analysis_results, self.audio_duration)
    
    def update_binaural_mood_combo(self):
        self.binaural_mood_combo.clear()
        mood_options = [TR.get_string("calmness"), TR.get_string("focus"), TR.get_string("happiness")]
        self.binaural_mood_combo.addItems(mood_options)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, TR.get_string("select_file_title"), "", "Audio Files (*.mp3 *.wav *.flac *.ogg *.aiff)")
        if file_path:
            self.file_path = file_path
            self.file_label.setText(os.path.basename(self.file_path))
            self.btn_analyze.setEnabled(True)
            self.report_text.clear()
            self.timeline_widget.set_data([], 0)
    
    def start_analysis(self):
        if not self.file_path:
            return
        
        channels_to_analyze = []
        if self.checkbox_left.isChecked():
            channels_to_analyze.append('left')
        if self.checkbox_right.isChecked():
            channels_to_analyze.append('right')

        if not channels_to_analyze:
            QMessageBox.warning(self, TR.get_string("error_analysis"), TR.get_string("no_channels_selected_error"))
            return

        self.btn_analyze.setEnabled(False)
        self.file_label.setText(f"{TR.get_string('processing')} {os.path.basename(self.file_path)}")
        self.report_text.clear()
        self.timeline_widget.set_data([], 0)
        self.analysis_thread = AnalysisThread(self.file_path, channels_to_analyze)
        self.analysis_thread.progress_signal.connect(self.update_progress)
        self.analysis_thread.results_signal.connect(self.display_results)
        self.analysis_thread.error_signal.connect(self.show_error)
        self.analysis_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def display_results(self, results, duration):
        self.analysis_results = results
        self.audio_duration = duration
        self.generate_report()
        self.timeline_widget.set_data(self.analysis_results, self.audio_duration)
        self.btn_analyze.setEnabled(True)
        self.file_label.setText(f"{TR.get_string('completed')} {os.path.basename(self.file_path)}")
        self.progress_bar.setValue(0)

    def show_error(self, error_message):
        QMessageBox.critical(self, TR.get_string("error_analysis"), f"{TR.get_string('error_message')} {error_message}")
        self.btn_analyze.setEnabled(True)
        self.file_label.setText(TR.get_string('error_retry'))
        self.progress_bar.setValue(0)
    
    def get_danger_color(self, level):
        if level >= 4: return QColor("#ef4444")
        if level == 3: return QColor("#f97316")
        if level == 2: return QColor("#fde047")
        return QColor("#22c55e")

    def get_danger_text(self, level):
        if level >= 4: return TR.get_string("danger_very_high")
        if level == 3: return TR.get_string("danger_high")
        if level == 2: return TR.get_string("danger_medium")
        return TR.get_string("danger_low")

    def generate_report(self):
        report_content = f"<h3 style='color:#a7f3d0;'><b>{TR.get_string('exec_report')}</b></h3>"
        report_content += "<hr style='border: 1px solid #374151;'><br>"
        report_content += f"<span style='color:#a7f3d0;'><b>{TR.get_string('audio_file')}</b></span> {os.path.basename(self.file_path) if self.file_path else 'N/A'}<br>"
        
        minutes = int(self.audio_duration // 60)
        seconds = self.audio_duration % 60
        duration_str = f"{minutes} {TR.get_string('min')} {seconds:.2f} {TR.get_string('sec')}"
        report_content += f"<span style='color:#a7f3d0;'><b>{TR.get_string('duration')}</b></span> {duration_str}<br><br>"
        
        if not self.analysis_results:
            report_content += f"<p>{TR.get_string('no_effects')}</p>"
            self.report_text.setHtml(report_content)
            return

        left_results = [r for r in self.analysis_results if r['channel'] == 'left']
        right_results = [r for r in self.analysis_results if r['channel'] == 'right']
        mono_results = [r for r in self.analysis_results if r['channel'] == 'mono']
        
        if left_results:
            report_content += f"<h3 style='color:#a7f3d0;'><b>{TR.get_string('left_channel_report')}</b></h3><hr style='border: 1px solid #374151;'>"
            report_content += self.get_detailed_channel_report(left_results)
        if right_results:
            report_content += f"<h3 style='color:#a7f3d0;'><b>{TR.get_string('right_channel_report')}</b></h3><hr style='border: 1px solid #374151;'>"
            report_content += self.get_detailed_channel_report(right_results)
        if mono_results:
            report_content += f"<h3 style='color:#a7f3d0;'><b>{TR.get_string('left_channel_report')}</b></h3><hr style='border: 1px solid #374151;'>"
            report_content += self.get_detailed_channel_report(mono_results)
        
        self.report_text.setHtml(report_content)

    def get_detailed_channel_report(self, results):
        if not results:
            return f"<p><i>{TR.get_string('no_effects')}</i></p>"
            
        report = ""
        for i, result in enumerate(results):
            mood_key = result['mood']
            
            # Use the standardized English keys to get data from FREQUENCY_DATA
            mood_data = FREQUENCY_DATA.get(mood_key, FREQUENCY_DATA["unidentified"])
            
            danger_color = self.get_danger_color(mood_data["danger_level"]).name()
            danger_text = self.get_danger_text(mood_data["danger_level"])
            
            report += f"<p style='color:#d1d5db;'><b>{TR.get_string('segment_report_title')}</b><br>"
            report += f"<b>{TR.get_string('time_range')}</b> {TR.get_string('from')} <span style='color: #6ee7b7;'>{result['start_time']:.2f}</span> {TR.get_string('to')} <span style='color: #6ee7b7;'>{result['end_time']:.2f}</span> {TR.get_string('sec')}<br>"
            report += f"<b>{TR.get_string('mood_detected')}</b> <span style='color: {mood_data['color']};'><b>{TR.get_string(mood_key)}</b></span><br>"
            report += f"<b>{TR.get_string('wave_type')}</b> {mood_data['wave_type'][TR.lang]}<br>"
            report += f"<b>{TR.get_string('physical_effect')}</b> {mood_data['physical_effect'][TR.lang]}<br>"
            report += f"<b>{TR.get_string('mental_effect')}</b> {mood_data['mental_effect'][TR.lang]}<br>"
            report += f"<b>{TR.get_string('hormonal_effect')}</b> {mood_data['hormonal_effect'][TR.lang]}<br>"
            report += f"<b>{TR.get_string('danger_level')}</b> <span style='color:{danger_color}'><b>{danger_text}</b></span><br><br>"
            report += "</p>"
        return report

    def start_mic_analysis(self):
        if self.realtime_thread and self.realtime_thread.isRunning():
            self.stop_mic_analysis()
        self.btn_mic_start.setEnabled(False)
        self.btn_mic_stop.setEnabled(True)
        self.report_text.clear()
        self.timeline_widget.set_data([], 0)
        self.realtime_thread = RealtimeAnalysisThread()
        self.realtime_thread.results_signal.connect(self.update_realtime_results)
        self.realtime_thread.start()

    def update_realtime_results(self, results, duration):
        self.analysis_results = results
        self.audio_duration = duration
        self.timeline_widget.set_data(self.analysis_results, self.audio_duration)
        self.generate_report()

    def stop_mic_analysis(self):
        if self.realtime_thread:
            self.realtime_thread.stop()
            self.realtime_thread.wait()
            self.realtime_thread = None
        self.btn_mic_start.setEnabled(True)
        self.btn_mic_stop.setEnabled(False)

    def start_binaural_generator(self):
        if self.binaural_thread and self.binaural_thread.isRunning():
            self.binaural_thread.stop_playback()
        self.btn_binaural_start.setEnabled(False)
        self.btn_binaural_stop.setEnabled(True)
        selected_mood_str = self.binaural_mood_combo.currentText()
        
        # Mapping translated mood string back to the standardized English key
        mood_key_map = {v: k for k, v in TRANSLATION[TR.lang].items() if k in BINAURAL_DATA}
        selected_mood_key = mood_key_map.get(selected_mood_str)
        
        if selected_mood_key:
            self.binaural_thread = BinauralGeneratorThread(selected_mood_key)
            self.binaural_thread.stop_signal.connect(lambda: (self.btn_binaural_start.setEnabled(True), self.btn_binaural_stop.setEnabled(False)))
            self.binaural_thread.start_playback()

    def stop_binaural_generator(self):
        if self.binaural_thread:
            self.binaural_thread.stop_playback()
        self.btn_binaural_stop.setEnabled(False)
        self.btn_binaural_start.setEnabled(True)

# Helper classes for threading (RealtimeAnalysisThread & BinauralGeneratorThread) remain the same
class RealtimeAnalysisThread(QThread):
    results_signal = pyqtSignal(list, float)
    error_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.weights = {'freq': 10, 'tempo': 10, 'amp': 10}
        self._is_running = True
        self.results = []
        self.sr = 44100
        self.chunk_size = 2048
        self.time_elapsed = 0

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            with sd.InputStream(samplerate=self.sr, channels=2, blocksize=self.chunk_size) as stream:
                while self._is_running:
                    data, overflowed = stream.read(self.chunk_size)
                    if overflowed: print("Buffer overflow!")
                    
                    left_channel_data = data[:, 0]
                    right_channel_data = data[:, 1]
                    
                    analysis_results = []
                    
                    dominant_freq_left = np.abs(np.fft.fftfreq(len(left_channel_data), 1/self.sr)[np.argmax(np.abs(np.fft.fft(left_channel_data)))])
                    tempo_left = librosa.beat.beat_track(y=left_channel_data, sr=self.sr, units='time', hop_length=512)[0]
                    rms_left = np.mean(librosa.feature.rms(y=left_channel_data))
                    dominant_mood_left = self.determine_mood(dominant_freq_left, tempo_left, rms_left)
                    analysis_results.append({"channel": "left", "start_time": self.time_elapsed, "end_time": self.time_elapsed + self.chunk_size/self.sr, "mood": dominant_mood_left})

                    dominant_freq_right = np.abs(np.fft.fftfreq(len(right_channel_data), 1/self.sr)[np.argmax(np.abs(np.fft.fft(right_channel_data)))])
                    tempo_right = librosa.beat.beat_track(y=right_channel_data, sr=self.sr, units='time', hop_length=512)[0]
                    rms_right = np.mean(librosa.feature.rms(y=right_channel_data))
                    dominant_mood_right = self.determine_mood(dominant_freq_right, tempo_right, rms_right)
                    analysis_results.append({"channel": "right", "start_time": self.time_elapsed, "end_time": self.time_elapsed + self.chunk_size/self.sr, "mood": dominant_mood_right})

                    self.results.extend(analysis_results)
                    self.time_elapsed += self.chunk_size / self.sr
                    consolidated_results = self.consolidate_results(self.results)
                    self.results_signal.emit(consolidated_results, self.time_elapsed)
        except Exception as e:
            self.error_signal.emit(str(e))
    
    def determine_mood(self, dominant_freq, tempo, mean_amplitude):
        if mean_amplitude < 0.01:
            return "silence"
        
        mood_scores = {}
        for mood_key, data in FREQUENCY_DATA.items():
            if mood_key == "silence" or mood_key == "unidentified": continue
            score = 0
            for freq_range in data["frequency_ranges"]:
                if freq_range[0] <= dominant_freq <= freq_range[1]: score += self.weights['freq']; break
            if data["tempo_range"][0] <= tempo <= data["tempo_range"][1]: score += self.weights['tempo']
            if data["amplitude_range"][0] <= mean_amplitude <= data["amplitude_range"][1]: score += self.weights['amp']
            mood_scores[mood_key] = score
        if mood_scores and max(mood_scores.values()) > 0:
            return max(mood_scores, key=mood_scores.get)
        return "unidentified"
    
    def consolidate_results(self, results):
        if not results:
            return []
            
        consolidated = []
        if not results:
            return consolidated

        left_results = [r for r in results if r['channel'] == 'left']
        right_results = [r for r in results if r['channel'] == 'right']
        mono_results = [r for r in results if r['channel'] == 'mono']
        
        if left_results:
            consolidated.extend(self._consolidate_channel(left_results))
        if right_results:
            consolidated.extend(self._consolidate_channel(right_results))
        if mono_results:
            consolidated.extend(self._consolidate_channel(mono_results))

        return consolidated
        
    def _consolidate_channel(self, results):
        if not results:
            return []
            
        consolidated = []
        current_segment = results[0]
        
        for i in range(1, len(results)):
            next_segment = results[i]
            if next_segment['mood'] == current_segment['mood']:
                current_segment['end_time'] = next_segment['end_time']
            else:
                consolidated.append(current_segment)
                current_segment = next_segment
                
        consolidated.append(current_segment)
        return consolidated


class BinauralGeneratorThread(QThread):
    stop_signal = pyqtSignal()
    
    def __init__(self, mood_key):
        super().__init__()
        self.mood_key = mood_key
        self._is_playing = False
        self.sr = 44100
        self.duration_sec = 600

    def start_playback(self):
        self._is_playing = True
        self.start()

    def stop_playback(self):
        self._is_playing = False
        self.wait()
        sd.stop()
        self.stop_signal.emit()

    def run(self):
        try:
            data = BINAURAL_DATA.get(self.mood_key)
            if not data: return
            
            freq1, freq2 = data["freq1"], data["freq2"]
            duration = data["duration"]
            t = np.linspace(0., duration, int(self.sr * duration), endpoint=False)
            
            wave1 = 0.5 * np.sin(2. * np.pi * freq1 * t)
            wave2 = 0.5 * np.sin(2. * np.pi * freq2 * t)
            
            stereo_wave = np.vstack((wave1, wave2)).T
            sd.play(stereo_wave, self.sr)
            while sd.get_stream().active and self._is_playing:
                sd.sleep(100)
            sd.stop()
        except Exception as e:
            print(f"Binaural playback error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuroTuneApp()
    window.show()
    sys.exit(app.exec())
