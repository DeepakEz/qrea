"""
Comprehensive Logging System for QREA
Supports console, file, TensorBoard, and Weights & Biases logging
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import json
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available. Install with: pip install wandb")


class ColoredFormatter(logging.Formatter):
    """Colored console logging formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class QREALogger:
    """Comprehensive logging system for QREA experiments"""
    
    def __init__(
        self,
        name: str,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # TensorBoard
        self.tensorboard_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = self.log_dir / 'tensorboard'
            tb_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = SummaryWriter(str(tb_dir))
            self.logger.info(f"TensorBoard logging to: {tb_dir}")
        
        # Weights & Biases
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if wandb_project is None:
                wandb_project = "qrea-warehouse"
            
            wandb.init(
                project=wandb_project,
                name=name,
                config=wandb_config or {},
                dir=str(self.log_dir)
            )
            self.logger.info(f"W&B logging to project: {wandb_project}")
        
        # Metrics history for JSON export
        self.metrics_history = []
        self.step_count = 0
        
    def debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)
    
    def critical(self, msg: str):
        """Log critical message"""
        self.logger.critical(msg)
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value"""
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(tag, value, step)
        
        # W&B
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
        
        # History
        self.metrics_history.append({
            'step': step,
            'tag': tag,
            'value': float(value),
            'timestamp': datetime.now().isoformat()
        })
    
    def log_scalars(self, tag_group: str, tag_value_dict: Dict[str, float], 
                   step: Optional[int] = None):
        """Log multiple scalars under a group"""
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalars(tag_group, tag_value_dict, step)
        
        # W&B
        if self.use_wandb:
            wandb.log({f"{tag_group}/{k}": v for k, v in tag_value_dict.items()}, step=step)
        
        # Individual scalars for history
        for tag, value in tag_value_dict.items():
            self.metrics_history.append({
                'step': step,
                'tag': f"{tag_group}/{tag}",
                'value': float(value),
                'timestamp': datetime.now().isoformat()
            })
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """Log histogram of values"""
        if step is None:
            step = self.step_count
        
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_histogram(tag, values, step)
        
        # W&B
        if self.use_wandb:
            wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        """Log an image"""
        if step is None:
            step = self.step_count
        
        # TensorBoard expects CHW format
        if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
            image = np.transpose(image, (2, 0, 1))
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_image(tag, image, step)
        
        # W&B
        if self.use_wandb:
            # W&B expects HWC format
            if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))
            wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_figure(self, tag: str, figure, step: Optional[int] = None):
        """Log a matplotlib figure"""
        if step is None:
            step = self.step_count
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_figure(tag, figure, step)
        
        # W&B
        if self.use_wandb:
            wandb.log({tag: figure}, step=step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text"""
        if step is None:
            step = self.step_count
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(tag, text, step)
        
        # W&B
        if self.use_wandb:
            wandb.log({tag: text}, step=step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration"""
        self.info("Configuration:")
        self.info(json.dumps(config, indent=2))
        
        # Save to file
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # W&B
        if self.use_wandb:
            wandb.config.update(config)
    
    def log_model_summary(self, model: torch.nn.Module):
        """Log model architecture summary"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = f"Model Summary:\n"
        summary += f"Total parameters: {total_params:,}\n"
        summary += f"Trainable parameters: {trainable_params:,}\n"
        summary += f"Architecture:\n{model}"
        
        self.info(summary)
        
        # Save to file
        summary_file = self.log_dir / "model_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
    
    def log_metrics_dict(self, metrics: Dict[str, float], prefix: str = "", 
                        step: Optional[int] = None):
        """Log a dictionary of metrics"""
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                tag = f"{prefix}/{key}" if prefix else key
                self.log_scalar(tag, float(value), step)
            elif isinstance(value, dict):
                # Recursive for nested dicts
                new_prefix = f"{prefix}/{key}" if prefix else key
                self.log_metrics_dict(value, new_prefix, step)
    
    def save_metrics_history(self):
        """Save metrics history to JSON"""
        history_file = self.log_dir / "metrics_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.info(f"Saved metrics history to: {history_file}")
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """Watch model parameters (W&B only)"""
        if self.use_wandb:
            wandb.watch(model, log_freq=log_freq)
    
    def save_artifact(self, file_path: str, artifact_type: str = "model",
                     name: Optional[str] = None):
        """Save artifact (W&B only)"""
        if self.use_wandb:
            if name is None:
                name = Path(file_path).stem
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
    
    def finish(self):
        """Close all loggers"""
        self.info("Finishing logging...")
        
        # Save metrics history
        self.save_metrics_history()
        
        # Close TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            self.info("TensorBoard writer closed")
        
        # Close W&B
        if self.use_wandb:
            wandb.finish()
            self.info("W&B run finished")
        
        self.info("Logging complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish()


def create_logger(
    experiment_name: str,
    base_dir: str = "logs",
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    config: Optional[Dict] = None
) -> QREALogger:
    """
    Factory function to create a configured logger
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for logs
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use W&B
        wandb_project: W&B project name
        config: Configuration dictionary
    
    Returns:
        Configured QREALogger instance
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    
    logger = QREALogger(
        name=experiment_name,
        log_dir=str(log_dir),
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=config
    )
    
    if config is not None:
        logger.log_config(config)
    
    return logger
