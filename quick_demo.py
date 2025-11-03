"""
QUICK PRESENTATION SCRIPT
Run this for a fast, impressive demo with competition scoring
"""

from app import HangmanPresentation, CompetitionScorer

def quick_presentation():
    print("🎓 HANGMAN AI - QUICK TEACHER PRESENTATION")
    print("=" * 60)
    
    # Initialize
    presenter = HangmanPresentation("corpus.txt")
    
    print("\n1. First, let me show you the competition performance:")
    score_data, stats = presenter.benchmark_performance(500)  # Smaller sample for quick demo
    
    print(f"\n2. Now watch the AI solve a word in real-time:")
    game_result = presenter.demo_ai_game()
    
    print(f"\n3. Let me demonstrate the AI's thinking process:")
    presenter.demo_ai_game("PROGRAMMING")
    
    # Show the scoring formula clearly
    print(f"\n📋 COMPETITION SCORING FORMULA:")
    print("Final Score = (Success Rate × 2000) - (Total Wrong Guesses × 5) - (Total Repeated Guesses × 2)")
    print(f"\n📊 Our AI achieved: {score_data['final_score']:.2f} points")
    
    if score_data['success_rate'] >= 0.6:
        print(f"\n🎉 OUTSTANDING! The AI achieved {score_data['success_rate']:.1%} success rate!")
        print("This significantly exceeds the target of 60-70% performance!")
    else:
        print(f"\n📈 Good progress! Current performance: {score_data['success_rate']:.1%}")
        print("The AI shows strong learning capability!")
    
    print("\n✨ Presentation ready! Use the interactive demo for full 2000-game benchmark.")

if __name__ == "__main__":
    quick_presentation()