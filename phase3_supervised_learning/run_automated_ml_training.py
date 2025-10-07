from ml_model_training import MLAddressMatchingTrainer

def run_automated_training():
    """Run automated ML training with quick mode."""
    print("🚀 Phase 3.4: Automated ML Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = MLAddressMatchingTrainer()
    
    # Run quick training
    print("\n⚡ Running Quick Training Mode (for fast results)...")
    report, saved_models = trainer.run_complete_training_pipeline(quick_tuning=True)
    
    print(f"\n🎉 Automated Training Complete!")
    print(f"   🏆 Best Model: {report['best_overall_model']['model_name']}")
    print(f"   🎯 Best F1 Score: {report['best_overall_model']['f1']:.4f}")
    print(f"   📊 Best ROC-AUC: {report['best_overall_model']['roc_auc']:.4f}")
    print(f"   📁 Models and results saved")
    
    return report, saved_models

if __name__ == "__main__":
    run_automated_training()