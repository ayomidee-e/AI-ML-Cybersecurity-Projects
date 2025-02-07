import pandas as pd
import numpy as np
import time
import threading
import queue
import logging
from datetime import datetime
import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from collections import deque
import json
import warnings
import psutil
import socket
import csv
from pathlib import Path

from main import AdvancedNetworkAnomalyDetector


class RealTimeAnalysisEngine:
    def __init__(self, model, processor, buffer_size=1000):
        self.model = model
        self.processor = processor
        self.packet_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        self.buffer_size = buffer_size
        self.running = False
        self.packet_buffer = deque(maxlen=buffer_size)
        self.anomaly_scores = deque(maxlen=buffer_size)
        self.last_bytes = psutil.net_io_counters()
        self.interfaces = psutil.net_if_stats()
        self.connections_file = 'network_connections.csv'
        self.traffic_file = 'network_traffic.csv'
        self.setup_logging()
        self.setup_csv_files()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('network_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_csv_files(self):
        # Setting-up connections CSV file
        if not Path(self.connections_file).exists():
            with open(self.connections_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'local_ip', 'local_port',
                                 'remote_ip', 'remote_port', 'status', 'type'])

        # Setting-up traffic CSV file
        if not Path(self.traffic_file).exists():
            with open(self.traffic_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'bytes_sent', 'bytes_recv',
                                 'packets_sent', 'packets_recv', 'total_bytes'])

    def save_connection(self, connection_info):
        with open(self.connections_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                connection_info['timestamp'],
                connection_info['local_ip'],
                connection_info['local_port'],
                connection_info['remote_ip'],
                connection_info['remote_port'],
                connection_info['status'],
                connection_info['type']
            ])

    def save_traffic(self, traffic_info):
        with open(self.traffic_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                traffic_info['timestamp'],
                traffic_info['bytes_sent'],
                traffic_info['bytes_recv'],
                traffic_info['packets_sent'],
                traffic_info['packets_recv'],
                traffic_info['total_bytes']
            ])

    def capture_network_stats(self):
        """Capturing network statistics and connections using psutil"""
        self.running = True

        while self.running:
            try:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Getting current network counters
                current = psutil.net_io_counters()

                # Calculate bytes sent/received since last check
                bytes_sent = current.bytes_sent - self.last_bytes.bytes_sent
                bytes_recv = current.bytes_recv - self.last_bytes.bytes_recv

                # Update last_bytes for next iteration
                self.last_bytes = current

                # Create traffic info dictionary
                traffic_info = {
                    'timestamp': current_time,
                    'bytes_sent': bytes_sent,
                    'bytes_recv': bytes_recv,
                    'packets_sent': current.packets_sent,
                    'packets_recv': current.packets_recv,
                    'total_bytes': bytes_sent + bytes_recv,
                }

                # Save traffic info
                self.save_traffic(traffic_info)
                self.packet_queue.put(traffic_info)

                # Capture and save connection information
                for conn in psutil.net_connections(kind='inet'):
                    try:
                        if conn.laddr and conn.raddr:  # Only save established connections
                            connection_info = {
                                'timestamp': current_time,
                                'local_ip': conn.laddr.ip,
                                'local_port': conn.laddr.port,
                                'remote_ip': conn.raddr.ip,
                                'remote_port': conn.raddr.port,
                                'status': conn.status,
                                'type': 'TCP' if conn.type == socket.SOCK_STREAM else 'UDP'
                            }
                            self.save_connection(connection_info)
                    except (AttributeError, ValueError) as e:
                        continue  # Skip incomplete connections

                time.sleep(1)  # Update every second

            except Exception as e:
                self.logger.error(f"Network capture error: {str(e)}")
                time.sleep(1)


    def process_packets(self):
        """Process captured network statistics and detect anomalies"""
        while self.running:
            try:
                packets = []
                while not self.packet_queue.empty() and len(packets) < 100:
                    packets.append(self.packet_queue.get())

                if packets:
                    # Convert packets to DataFrame
                    df = pd.DataFrame(packets)

                    # Add derived features
                    df['bytes_per_packet'] = (df['total_bytes'] /
                                              (df['packets_sent'] + df['packets_recv']).clip(lower=1))

                    # Store results
                    for packet in packets:
                        self.packet_buffer.append(packet)

                        # Anomaly detection based on threshold
                        is_anomaly = packet['total_bytes'] > 100  # 1MB threshold
                        self.anomaly_scores.append(float(is_anomaly))

                        if is_anomaly:
                            alert = {
                                'timestamp': packet['timestamp'],
                                'severity': 'HIGH',
                                'message': f'High network usage detected: {packet["total_bytes"] / 1024 / 1024:.2f} MB'
                            }
                            self.alert_queue.put(alert)
                            self.logger.warning(f"Anomaly detected: {json.dumps(alert)}")

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Processing error: {str(e)}")

    def start(self):
        """Start the real-time analysis engine"""
        self.running = True

        # Start network capture thread
        self.capture_thread = threading.Thread(
            target=self.capture_network_stats
        )
        self.capture_thread.start()

        # Start processing thread
        self.process_thread = threading.Thread(
            target=self.process_packets
        )
        self.process_thread.start()

        self.logger.info("Real-time analysis engine started")

    def stop(self):
        """Stop the real-time analysis engine"""
        self.running = False
        self.capture_thread.join()
        self.process_thread.join()
        self.logger.info("Real-time analysis engine stopped")


class DashboardApp:
    def __init__(self, analysis_engine):
        self.analysis_engine = analysis_engine
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Creating the dashboard layout"""
        self.app.layout = html.Div([
            html.H1('Network Traffic Monitor'),

            html.Div([
                html.Div([
                    html.H3('Real-time Network Traffic'),
                    dcc.Graph(id='traffic-graph'),
                ], className='six columns'),

                html.Div([
                    html.H3('Traffic Alerts'),
                    html.Div(id='alerts-div', style={'height': '400px', 'overflow-y': 'scroll'})
                ], className='six columns'),
            ], className='row'),

            html.Div([
                html.H3('Network Statistics'),
                dcc.Graph(id='stats-graph'),
            ]),

            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            )
        ])

    def setup_callbacks(self):
        @self.app.callback(
            [Output('traffic-graph', 'figure'),
             Output('stats-graph', 'figure'),
             Output('alerts-div', 'children')],
            [Input('interval-component', 'n_intervals')]
        )

        def update_graphs(n):
            # Debugging: Log the n_intervals and current traffic data
            print(f"Update interval: {n}")

            traffic_data = list(self.analysis_engine.packet_buffer)
            print(f"Traffic data length: {len(traffic_data)}")  # Check if data is being received

            if traffic_data:
                df = pd.DataFrame(traffic_data)

                # Putting the timestamp in a datetime format
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                traffic_fig = {
                    'data': [
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['bytes_sent'],
                            mode='lines',
                            name='Bytes Sent'
                        ),
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['bytes_recv'],
                            mode='lines',
                            name='Bytes Received'
                        )
                    ],
                    'layout': go.Layout(
                        title='Network Traffic Volume',
                        xaxis={'title': 'Time'},
                        yaxis={'title': 'Bytes'}
                    )
                }
            else:
                traffic_fig = {
                    'data': [
                        go.Scatter(
                            x=[datetime.now()],
                            y=[0],
                            mode='lines',
                            name='Bytes Sent'
                        )
                    ],
                    'layout': go.Layout(
                        title='Network Traffic Volume',
                        xaxis={'title': 'Time'},
                        yaxis={'title': 'Bytes'}
                    )
                }

            # Update statistics graph
            stats_fig = {
                'data': [
                    go.Bar(
                        x=['Sent', 'Received'],
                        y=[self.analysis_engine.last_bytes.bytes_sent / 1024 / 1024,
                           self.analysis_engine.last_bytes.bytes_recv / 1024 / 1024]
                    )
                ],
                'layout': go.Layout(
                    title='Total Traffic (MB)',
                    yaxis={'title': 'Megabytes'}
                )
            }

            # Update alerts
            alerts = []
            while not self.analysis_engine.alert_queue.empty():
                alert = self.analysis_engine.alert_queue.get()
                alerts.append(html.Div([
                    html.H4(f"Alert - {alert['severity']}"),
                    html.P(f"Time: {alert['timestamp']}"),
                    html.P(f"Message: {alert['message']}"),
                    html.Hr()
                ]))

            if not alerts:
                alerts = [html.P("No alerts detected.")]

            return traffic_fig, stats_fig, alerts

    def run(self, debug=False, port=8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=port)


def main():
    # Initialize real-time analysis engine
    detector = AdvancedNetworkAnomalyDetector()
    engine = RealTimeAnalysisEngine(detector, detector)

    print(f"Saving connection data to: {engine.connections_file}")
    print(f"Saving traffic data to: {engine.traffic_file}")

    # Start the engine
    engine.start()

    # Initialize and run dashboard
    dashboard = DashboardApp(engine)
    dashboard.run()


if __name__ == "__main__":
    main()
