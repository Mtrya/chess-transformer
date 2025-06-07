"""

"""

import gradio as gr
import chess
import chess.svg
import chess.pgn
import re
import torch
import os
import io
import math
from typing import Optional, Tuple, List
import traceback
from datetime import datetime

from utils import Engine, ChessformerConfig, StockfishConfig, UCI_MOVE_TO_IDX
from model import ChessFormerModel

from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

class ChessApp:
    def __init__(self, device):
        self.board = chess.Board()
        self.move_history = []
        self.current_engine = None
        self.analysis_engine = None
        self.game_over = False
        self.user_color = chess.WHITE
        self.models = {}
        self.device = device

        self.current_engine_eval = 0.0
        self.stockfish_eval = 0.0

        self.analysis_executor = ThreadPoolExecutor(max_workers=2)

        self.load_models()
        self.create_analysis_engine()

    def load_models(self):
        model_paths = {
            "ChessFormer-SL": "./ckpts/chessformer-sl_01.pth",
            "ChessFormer-RL": "./ckpts/chessformer-rl_final.pth"
        }

        for name, path in model_paths.items():
            if os.path.exists(path):
                print(f"Loading {name} from {path}...")
                checkpoint = torch.load(path,map_location=self.device)
                config = checkpoint["config"]
                model = ChessFormerModel(**config)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()

                self.models[name] = model
                print(f"Successfully loaded {name}.")
            else:
                print(f"Model file not found: {path}")

    def get_depth_limits(self, engine_type: str) -> Tuple[int,int]:
        if engine_type == "Stockfish":
            return 1,24,6
        else:
            return 0,6,0
        
    def create_evaluation_bar(self, eval_score: float, title: str) -> str:
        """Create HTML evaluation bar from user's perspective with page-matching colors"""
        # Convert eval_score from white's perspective to user's perspective
        user_eval = eval_score if self.user_color == chess.WHITE else -eval_score
        
        # Clamp evaluation between -1 and 1 for display
        clamped_eval = max(-1.0, min(1.0, user_eval))
        
        # Convert to percentage (0 = user losing, 100 = user winning)
        percentage = (clamped_eval + 1.0) / 2.0 * 100
        
        # Format evaluation text from user's perspective
        eval_text = f"{user_eval:+.2f}"
        if abs(user_eval) > 5:
            eval_text = "±∞" if user_eval > 0 else "∓∞"
        
        # Determine advantage text and colors (matching page theme)
        if user_eval > 0.5:
            advantage_text = "WINNING"
            text_color = "#1e40af"  # Blue-700
            indicator_color = "#3b82f6"  # Blue-500
        elif user_eval > 0.1:
            advantage_text = "SLIGHT ADVANTAGE"
            text_color = "#1e40af"
            indicator_color = "#60a5fa"  # Blue-400
        elif user_eval < -0.5:
            advantage_text = "LOSING"
            text_color = "#7c2d12"  # Orange-800 (more muted than red)
            indicator_color = "#ea580c"  # Orange-600
        elif user_eval < -0.1:
            advantage_text = "SLIGHT DISADVANTAGE"
            text_color = "#9a3412"  # Orange-700
            indicator_color = "#f97316"  # Orange-500
        else:
            advantage_text = "EQUAL POSITION"
            text_color = "#4b5563"  # Gray-600
            indicator_color = "#6b7280"  # Gray-500
        
        return f"""
        <div style="margin: 10px 0; font-family: 'Segoe UI', Arial, sans-serif;">
            <h4 style="margin: 5px 0 10px 0; color: #374151; font-size: 14px; font-weight: 600;">{title}</h4>
            
            <!-- Evaluation bar with page-matching gradient -->
            <div style="width: 100%; height: 40px; border: 2px solid #d1d5db; border-radius: 8px; position: relative; 
                        background: linear-gradient(to right, 
                            #fed7aa 0%,     /* Orange-200 - losing */
                            #fde68a 20%,    /* Yellow-200 */
                            #e5e7eb 50%,    /* Gray-200 - equal */
                            #bfdbfe 80%,    /* Blue-200 */
                            #93c5fd 100%    /* Blue-300 - winning */
                        ); 
                        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);">
                
                <!-- Evaluation indicator -->
                <div style="position: absolute; left: {percentage}%; top: 50%; transform: translateX(-50%) translateY(-50%); 
                            background: {indicator_color}; border: 3px solid white; border-radius: 50%; width: 18px; height: 18px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.15), 0 0 0 1px #d1d5db; z-index: 10;
                            transition: all 0.3s ease;"></div>
            </div>
            
            <!-- Evaluation text -->
            <div style="text-align: center; margin-top: 8px; padding: 8px; background: #f9fafb; 
                        border-radius: 6px; border: 1px solid #e5e7eb;">
                <div style="font-weight: 600; color: {text_color}; font-size: 16px; margin-bottom: 2px;">
                    {eval_text}
                </div>
                <div style="font-size: 10px; color: {text_color}; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500; opacity: 0.8;">
                    {advantage_text}
                </div>
            </div>
        </div>
        """

    def create_analysis_engine(self):
        """Create optimized Stockfish depth 27 engine for analysis"""
        try:
            config = StockfishConfig(
                engine_path="/usr/games/stockfish",
                depth=27
            )
            self.analysis_engine = Engine(type="stockfish", stockfish_config=config)
            
            # Configure Stockfish for faster analysis
            if self.analysis_engine and hasattr(self.analysis_engine, 'engine_path'):
                # We'll patch the engine creation to use optimized settings
                pass
            
            print("Analysis engine (Stockfish depth 27) created successfully")
        except Exception as e:
            print(f"Failed to create analysis engine: {e}")
            self.analysis_engine = None

    def update_evaluations(self):
        """Update evaluations from both engines with optimized Stockfish analysis"""
        # Get current engine evaluation
        if self.current_engine:
            try:
                self.current_engine_eval = self.current_engine.analyze_position(self.board.copy())
                if self.current_engine_eval is None:
                    self.current_engine_eval = 0.0
            except:
                self.current_engine_eval = 0.0
        
        # Get optimized Stockfish analysis
        if self.analysis_engine:
            try:
                self.stockfish_eval = self.fast_stockfish_analysis(self.board.copy())
                if self.stockfish_eval is None:
                    self.stockfish_eval = 0.0
            except:
                self.stockfish_eval = 0.0

    def update_evaluations_async(self):
        """Update evaluations asynchronously"""
        def update_current_engine():
            if self.current_engine:
                try:
                    self.current_engine_eval = self.current_engine.analyze_position(self.board.copy())
                    if self.current_engine_eval is None:
                        self.current_engine_eval = 0.0
                except:
                    self.current_engine_eval = 0.0
        
        def update_stockfish():
            try:
                self.stockfish_eval = self.fast_stockfish_analysis(self.board.copy())
                if self.stockfish_eval is None:
                    self.stockfish_eval = 0.0
            except:
                self.stockfish_eval = 0.0
        
        # Run both analyses in parallel
        future1 = self.analysis_executor.submit(update_current_engine)
        future2 = self.analysis_executor.submit(update_stockfish)
        
        # Wait for both to complete
        future1.result()
        future2.result()

    def fast_stockfish_analysis(self, board: chess.Board) -> Optional[float]:
        """Fast Stockfish analysis with optimized settings"""
        try:
            import chess.engine
            
            # Create engine with optimized settings
            with chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish") as engine:
                # Configure for speed
                engine.configure({
                    "Threads": min(8, os.cpu_count() or 4),  # Use multiple threads
                    "Hash": 256,  # 256MB hash table
                    "UCI_AnalyseMode": True
                })
                
                # Use time limit instead of depth for faster analysis
                info = engine.analyse(
                    board, 
                    chess.engine.Limit(time=1.0),  # 1 second analysis
                )
                
                score_obj = info.get("score")
                if score_obj is None:
                    return None
                    
                pov_score = score_obj.pov(chess.WHITE)
                
                if pov_score.is_mate():
                    mate_score = pov_score.mate()
                    cp = 10000.0 if mate_score > 0 else -10000.0
                elif pov_score.cp is not None:
                    cp = float(pov_score.cp)
                else:
                    return None
                
                # Normalize score
                normalized_score = 2 / (1 + math.exp(-0.004 * cp)) - 1
                return normalized_score
                
        except Exception as e:
            print(f"Fast Stockfish analysis error: {e}")
            return None
        
    def create_engine(self, engine_type: str, depth: int, temperature: float=0.5) -> Optional[Engine]:
        if engine_type == "Stockfish":
            config = StockfishConfig(
                engine_path="/usr/games/stockfish",
                depth=depth
            )
            return Engine(type="stockfish",stockfish_config=config)
        elif engine_type in self.models:
            config = ChessformerConfig(
                chessformer=self.models[engine_type],
                device=self.device,
                temperature=temperature,
                depth=depth if depth > 0 else 0,
                top_k=8,
                decay_rate=0.6,
                max_batch_size=800
            )
            return Engine(type="chessformer",chessformer_config=config)
        
        return None
    
    def parse_move(self, move_str: str) -> Optional[chess.Move]:
        """Parse move input in either UCI format ("e2e4") or algebraic notation ("Ne5")"""
        if not move_str:
            return None
        
        move_str = move_str.strip()

        # Try UCI format first
        uci_pattern = r'^[a-h][1-8][a-h][1-8][qrbn]?$'
        if re.match(uci_pattern,move_str.lower()):
            try:
                return chess.Move.from_uci(move_str.lower())
            except ValueError:
                pass

        # Try algrebraic notation
        try:
            return self.board.parse_san(move_str)
        except ValueError:
            pass

        return None
    
    def get_board_svg(self) -> str:
        """Generate SVG representation of the chess board"""
        flip = (self.user_color == chess.BLACK)

        lastmove = None
        if self.move_history:
            lastmove = self.move_history[-1]
        
        svg = chess.svg.board(
            board=self.board,
            flipped=flip,
            lastmove=lastmove,
            size=600
        )
        return svg
    
    def get_move_history_text(self) -> str:
        """Generate move history in PGN format"""
        try:
            game = chess.pgn.Game()
            game.headers["Event"] = "ChessFormer Demo"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["White"] = "You" if self.user_color == chess.WHITE else "Engine"
            game.headers["Black"] = "Engine" if self.user_color == chess.WHITE else "You"
            
            node = game
            temp_board = chess.Board()

            for move in self.move_history:
                node = node.add_variation(move)
                temp_board.push(move)

            if self.game_over:
                outcome = self.board.outcome()
                if outcome:
                    if outcome.winner == chess.WHITE:
                        game.headers["Result"] = "1-0"
                    elif outcome.winner == chess.BLACK:
                        game.headers["Result"] = "0-1"
                    else:
                        game.headers["Result"] = "1/2-1/2"
                else:
                    game.headers["Result"] = "*"
            else:
                game.headers["Result"] = "*"

            return str(game)
        except Exception as e:
            print(f"Error generating move history: {e}")
            return "Move history unavailable"
        
    def export_pgn(self) -> str:
        return self.get_move_history_text()
    
    def import_fen(self, fen: str) -> Tuple[str,str,str,str,str]:
        try:
            test_board = chess.Board(fen.strip())
            self.board = test_board
            self.move_history = []
            self.game_over = False
            self.update_evaluations()
            
            return (
                self.get_board_svg(),
                self.get_move_history_text(),
                f"Position loaded from FEN: {fen}",
                "",
                self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
                self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
            )
        except Exception as e:
            return (
                self.get_board_svg(),
                self.get_move_history_text(),
                f"Invalid FEN: {str(e)}",
                "",
                self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
                self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
            )
        
    def import_pgn(self, pgn_text: str) -> Tuple[str,str,str,str,str]:
        try:
            pgn_io = io.StringIO(pgn_text.strip())
            game = chess.pgn.read_game(pgn_io)
            
            if game is None:
                raise ValueError("Could not parse PGN")
            
            self.board = game.board()
            self.move_history = []
            
            for move in game.mainline_moves():
                self.board.push(move)
                self.move_history.append(move)
            
            self.game_over = self.board.is_game_over()
            self.update_evaluations()
            
            return (
                self.get_board_svg(),
                self.get_move_history_text(),
                f"Game loaded from PGN ({len(self.move_history)} moves)",
                "",
                self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
                self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
            )
        except Exception as e:
            return (
                self.get_board_svg(),
                self.get_move_history_text(),
                f"Invalid PGN: {str(e)}",
                "",
                self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
                self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
            )

    def make_user_move(self, move_str: str) -> Tuple[str,str,str,str,str,str]:
        if self.game_over:
            return (
                self.get_board_svg(),
                self.get_move_history_text(),
                "Game is over. Click 'New Game' to start a new game.",
                "",
                self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
                self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
            )
        
        if self.board.turn != self.user_color:
            return (
                self.get_board_svg(),
                self.get_move_history_text(),
                "It's not your turn now!",
                "",
                self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
                self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
            )
        
        move = self.parse_move(move_str)
        if move is None:
            return (
                self.get_board_svg(),
                self.get_move_history_text(),
                f"Invalid move: '{move_str}'. Try formats like 'e2e4' or 'Ne5'",
                "",
                self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
                self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
            )
        
        if move not in self.board.legal_moves:
            return (
                self.get_board_svg(),
                self.get_move_history_text(),
                f"Illegal move: '{move_str}'",
                "",
                self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
                self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
            )
        
        self.board.push(move)
        self.move_history.append(move)

        self.update_evaluations()

        if self.board.is_game_over():
            self.game_over = True
            outcome = self.board.outcome()
            if outcome:
                if outcome.winner == self.user_color:
                    status = "🎉🏆 CONGRATULATIONS! YOU WON! 🏆🎉"
                    status += f"\n🎯 Victory by {outcome.termination.name}! 🎯"
                elif outcome.winner is None:
                    status = "🤝 GAME DRAWN 🤝"
                    status += f"\n⚖️ Draw by {outcome.termination.name} ⚖️"
                else:
                    status = "😔 YOU LOST 😔"
                    status += f"\n💔 Defeated by {outcome.termination.name} 💔"
            else:
                status = "🏁 GAME OVER 🏁"
            
            return (
                self.get_board_svg(),
                self.get_move_history_text(),
                status,
                "",
                self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
                self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
            )
        
        # Get engine move
        try:
            engine_move_uci, engine_value = self.current_engine.move(self.board)
            
            if engine_move_uci == "<claim_draw>":
                self.game_over = True
                status = "Engine claimed a draw."
            else:
                engine_move = chess.Move.from_uci(engine_move_uci)
                self.board.push(engine_move)
                self.move_history.append(engine_move)

                if self.board.is_game_over():
                    self.game_over = True
                    outcome = self.board.outcome()
                    if outcome:
                        if outcome.winner == self.user_color:
                            status = "🎉🏆 CONGRATULATIONS! YOU WON! 🏆🎉"
                            status += f"\n🎯 Victory by {outcome.termination.name}! 🎯"
                        elif outcome.winner is None:
                            status = "🤝 GAME DRAWN 🤝"
                            status += f"\n⚖️ Draw by {outcome.termination.name} ⚖️"
                        else:
                            status = "😔 YOU LOST 😔"
                            status += f"\n💔 Defeated by {outcome.termination.name} 💔"
                    else:
                        status = "🏁 GAME OVER 🏁"
                else:
                    status = f"Engine played: {engine_move.uci()}."
            
        except Exception as e:
            status = f"Engine error: {str(e)}"
            print(f"Engine error: {e}")
            traceback.print_exc()

        return (
            self.get_board_svg(),
            self.get_move_history_text(),
            status,
            "", # clear input
            self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
            self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
        )
    
    def new_game(self, engine_type: str, depth: int, color: str, temperature: float) -> Tuple[str,str,str,str,str,str]:
        "Start a new game"
        self.board = chess.Board()
        self.move_history = []
        self.game_over = False
        self.user_color = chess.WHITE if color == "White" else chess.BLACK
        
        # Create new engine
        self.current_engine = self.create_engine(engine_type, depth, temperature)

        self.update_evaluations()
        
        if self.current_engine is None:
            status = f"Failed to create {engine_type} engine."
        else:
            status = f"New game started! You are playing {color} against {engine_type} (depth {depth})."
            
            # If user is black, make engine move first
            if self.user_color == chess.BLACK:
                try:
                    engine_move_uci, engine_value = self.current_engine.move(self.board)
                    if engine_move_uci != "<claim_draw>":
                        engine_move = chess.Move.from_uci(engine_move_uci)
                        self.board.push(engine_move)
                        self.move_history.append(engine_move)
                        status += f" Engine opened with: {engine_move.uci()}"
                except Exception as e:
                    status += f" Engine error on first move: {str(e)}"
        
        return (
            self.get_board_svg(),
            self.get_move_history_text(),
            status,
            "",
            self.create_evaluation_bar(self.stockfish_eval, "Stockfish Analysis (from your perspective)"),
            self.create_evaluation_bar(self.current_engine_eval, "Engine Analysis (from your perspective)")
        )
    

app = ChessApp(torch.device("cpu"))

def create_interface():
    """Create the Gradio interface with improved layout"""
    
    with gr.Blocks(title="ChessFormer Demo", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🏆 ChessFormer Demo")
        gr.Markdown("Play chess against ChessFormer models or Stockfish!")
        
        with gr.Row():
            # Left column - Analysis + History
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Position Analysis")
                
                # Stockfish Analysis
                stockfish_eval_display = gr.HTML(
                    value=app.create_evaluation_bar(0.0, "Stockfish Analysis"),
                    label="Stockfish"
                )
                
                # Current Engine Analysis  
                current_engine_eval_display = gr.HTML(
                    value=app.create_evaluation_bar(0.0, "Engine Analysis"),
                    label="Engine"
                )
                
                # Move history
                gr.Markdown("### 📝 Game History")
                history_display = gr.Textbox(
                    value=app.get_move_history_text(),
                    label="PGN",
                    lines=12,
                    max_lines=15,
                    interactive=False
                )
            
            # Middle column - Game Board + Controls
            with gr.Column(scale=4):
                # Chess board display
                board_display = gr.HTML(
                    value=app.get_board_svg(),
                    label="Chess Board"
                )
                
                # Move input
                with gr.Row():
                    move_input = gr.Textbox(
                        placeholder="Enter move (e.g., 'e2e4' or 'Ne5')",
                        label="Your Move",
                        scale=4
                    )
                    move_button = gr.Button("Make Move", variant="primary", scale=1)
                
                # Game status
                status_display = gr.Textbox(
                    value="Click 'New Game' to start playing!",
                    label="Game Status",
                    interactive=False,
                    lines=2
                )
            
            # Right column - Settings + Import/Export
            with gr.Column(scale=2):
                # Engine settings
                gr.Markdown("### ⚙️ Game Settings")
                
                engine_choices = ["Stockfish"] + list(app.models.keys())
                engine_select = gr.Dropdown(
                    choices=engine_choices,
                    value="ChessFormer-SL" if engine_choices else None,
                    label="Opponent Engine"
                )
                
                depth_slider = gr.Slider(
                    minimum=0,
                    maximum=6,
                    value=0,
                    step=1,
                    label="Engine Depth"
                )
                
                color_select = gr.Radio(
                    choices=["White", "Black"],
                    value="White",
                    label="Your Color"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    label="Temperature (ChessFormer only)"
                )
                
                new_game_button = gr.Button("🔄 New Game", variant="secondary", size="lg")
                
                # Import/Export section
                gr.Markdown("### 📁 Import/Export")
                
                with gr.Tabs():
                    with gr.Tab("Import FEN"):
                        fen_input = gr.Textbox(
                            placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                            label="FEN String",
                            lines=2
                        )
                        import_fen_button = gr.Button("Import FEN")
                    
                    with gr.Tab("Import PGN"):
                        pgn_input = gr.Textbox(
                            placeholder="1. e4 e5 2. Nf3 Nc6...",
                            label="PGN Text",
                            lines=3
                        )
                        import_pgn_button = gr.Button("Import PGN")
                    
                    with gr.Tab("Export"):
                        export_button = gr.Button("📁 Download PGN")
                        export_output = gr.File(label="Download")
        
        # Available models info
        gr.Markdown("### 🤖 Available Models")
        if app.models:
            model_info = "**Loaded ChessFormer models:**\n" + "\n".join([f"• {name}" for name in app.models.keys()])
        else:
            model_info = "⚠️ No ChessFormer models found. Make sure model checkpoints are in the ./ckpts/ directory."
        gr.Markdown(model_info)
        
        # Function to update depth limits based on engine selection
        def update_depth_limits(engine_type):
            min_depth, max_depth, value = app.get_depth_limits(engine_type)
            return gr.Slider(minimum=min_depth, maximum=max_depth, value=value, step=1)
        
        # Function to export PGN
        def export_pgn_file():
            pgn_content = app.export_pgn()
            filename = f"chess_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
            with open(filename, 'w') as f:
                f.write(pgn_content)
            return filename
        
        # Event handlers (same as before...)
        engine_select.change(
            fn=update_depth_limits,
            inputs=[engine_select],
            outputs=[depth_slider]
        )
        
        move_button.click(
            fn=app.make_user_move,
            inputs=[move_input],
            outputs=[board_display, history_display, status_display, move_input, 
                    stockfish_eval_display, current_engine_eval_display]
        )
        
        move_input.submit(
            fn=app.make_user_move,
            inputs=[move_input],
            outputs=[board_display, history_display, status_display, move_input,
                    stockfish_eval_display, current_engine_eval_display]
        )
        
        new_game_button.click(
            fn=app.new_game,
            inputs=[engine_select, depth_slider, color_select, temperature_slider],
            outputs=[board_display, history_display, status_display, move_input,
                    stockfish_eval_display, current_engine_eval_display]
        )
        
        import_fen_button.click(
            fn=app.import_fen,
            inputs=[fen_input],
            outputs=[board_display, history_display, status_display, fen_input,
                    stockfish_eval_display, current_engine_eval_display]
        )
        
        import_pgn_button.click(
            fn=app.import_pgn,
            inputs=[pgn_input],
            outputs=[board_display, history_display, status_display, pgn_input,
                    stockfish_eval_display, current_engine_eval_display]
        )
        
        export_button.click(
            fn=export_pgn_file,
            outputs=[export_output]
        )
        
        # Auto-start a new game when interface loads
        interface.load(
            fn=app.new_game,
            inputs=[gr.State("Stockfish"), gr.State(6), gr.State("White"), gr.State(0.5)],
            outputs=[board_display, history_display, status_display, move_input,
                    stockfish_eval_display, current_engine_eval_display]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    interface.launch()
