import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
import string
import re
from itertools import combinations

# ==================== ADVANCED AI WITH MULTIPLE STRATEGIES ====================

class AdvancedHangmanAI:
    def __init__(self):
        self.word_list = []
        self.word_patterns = defaultdict(list)
        self.letter_frequencies = None
        self.position_frequencies = None
        self.bigram_frequencies = defaultdict(Counter)
        self.common_words = set()
        self.word_frequencies = Counter()
        self.vowel_consonant_stats = None
        
    def train(self, corpus):
        """Advanced training with multiple linguistic features"""
        print("Training Advanced Hangman AI with enhanced features...")
        
        # Clean and process corpus
        self.word_list = []
        for word in corpus:
            clean_word = ''.join(c.upper() for c in word if c.isalpha())
            if 3 <= len(clean_word) <= 12 and clean_word.isalpha():
                self.word_list.append(clean_word)
        
        print(f"Training on {len(self.word_list)} valid words")
        
        # Build comprehensive pattern database
        self._build_enhanced_pattern_database()
        
        # Calculate advanced statistics
        self._calculate_comprehensive_frequencies()
        self._calculate_bigram_statistics()
        self._calculate_vowel_consonant_stats()
        
        # Identify common words with frequency
        self._identify_common_words_with_frequency()
        
        print("Advanced training completed!")
    
    def _build_enhanced_pattern_database(self):
        """Build comprehensive pattern database with word frequencies"""
        word_counter = Counter(self.word_list)
        self.word_frequencies = word_counter
        
        for word in self.word_list:
            length = len(word)
            # Store word with its frequency weight
            self.word_patterns[length].append((word, word_counter[word]))
    
    def _calculate_comprehensive_frequencies(self):
        """Calculate advanced frequency statistics"""
        all_text = ''.join(self.word_list)
        total_chars = len(all_text)
        
        # Overall frequency
        self.letter_frequencies = {}
        for letter in string.ascii_uppercase:
            count = all_text.count(letter)
            self.letter_frequencies[letter] = count / total_chars if total_chars > 0 else 0
        
        # Position frequency with smoothing
        max_len = max(len(word) for word in self.word_list) if self.word_list else 12
        self.position_frequencies = np.zeros((max_len, 26))
        
        for word in self.word_list:
            for pos, letter in enumerate(word):
                if pos < max_len:
                    idx = ord(letter) - ord('A')
                    self.position_frequencies[pos, idx] += 1
        
        # Add Laplace smoothing and normalize
        self.position_frequencies += 0.5
        self.position_frequencies /= self.position_frequencies.sum(axis=1, keepdims=True)
    
    def _calculate_bigram_statistics(self):
        """Calculate bigram (2-letter combination) frequencies"""
        for word in self.word_list:
            for i in range(len(word) - 1):
                bigram = word[i:i+2]
                self.bigram_frequencies[bigram[0]][bigram[1]] += 1
    
    def _calculate_vowel_consonant_stats(self):
        """Calculate vowel/consonant patterns"""
        vowels = 'AEIOU'
        self.vowel_consonant_stats = {
            'vowel_freq': sum(self.letter_frequencies.get(v, 0) for v in vowels),
            'common_vowels': ['E', 'A', 'I', 'O', 'U'],
            'common_consonants': ['T', 'N', 'S', 'R', 'H', 'L', 'D', 'C', 'M', 'W']
        }
    
    def _identify_common_words_with_frequency(self):
        """Identify most common words with frequency weighting"""
        self.common_words = set(word for word, count in self.word_frequencies.most_common(200))
    
    def get_possible_words(self, pattern, guessed_letters, wrong_guesses):
        """Get possible words with frequency weighting"""
        word_length = len(pattern)
        possible_words = []
        
        if word_length not in self.word_patterns:
            return []
        
        for word, frequency in self.word_patterns[word_length]:
            if self._matches_pattern(word, pattern, guessed_letters):
                # Weight by word frequency
                possible_words.append((word, frequency))
        
        return possible_words
    
    def _matches_pattern(self, word, pattern, guessed_letters):
        """Enhanced pattern matching"""
        # Check exact matches
        for i, (pattern_char, word_char) in enumerate(zip(pattern, word)):
            if pattern_char != '_':
                if pattern_char != word_char:
                    return False
            else:
                # Blank position shouldn't contain guessed letters
                if word_char in guessed_letters:
                    return False
                
                # Additional constraint: if we've guessed letters that must appear elsewhere
                for guessed in guessed_letters:
                    if guessed in word and guessed not in pattern:
                        # This guessed letter should appear in the word but isn't in pattern yet
                        # This means it should appear in one of the blank positions
                        if word.count(guessed) > pattern.count(guessed):
                            continue
                        else:
                            return False
        
        return True
    
    def calculate_letter_probabilities(self, pattern, guessed_letters, wrong_guesses):
        """Advanced probability calculation with multiple strategies"""
        word_length = len(pattern)
        possible_words = self.get_possible_words(pattern, guessed_letters, wrong_guesses)
        
        if not possible_words:
            return self._advanced_fallback(guessed_letters, pattern, wrong_guesses)
        
        # Strategy 1: Frequency-weighted letter probabilities
        freq_probs = self._frequency_weighted_probabilities(possible_words, guessed_letters)
        
        # Strategy 2: Position-based probabilities
        pos_probs = self._position_based_probabilities(pattern, guessed_letters)
        
        # Strategy 3: Bigram context probabilities
        bigram_probs = self._bigram_context_probabilities(pattern, guessed_letters)
        
        # Strategy 4: Game state adaptive weights
        state_weights = self._game_state_weights(wrong_guesses, len(possible_words))
        
        # Combine all strategies with dynamic weights
        combined_probs = (
            freq_probs * state_weights['frequency'] +
            pos_probs * state_weights['position'] +
            bigram_probs * state_weights['bigram']
        )
        
        # Apply vowel/consonant strategy
        combined_probs = self._vowel_consonant_strategy(combined_probs, guessed_letters, wrong_guesses)
        
        # Zero out guessed letters
        for letter in guessed_letters:
            if letter in string.ascii_uppercase:
                idx = ord(letter) - ord('A')
                combined_probs[idx] = 0
        
        # Normalize
        if combined_probs.sum() > 0:
            combined_probs /= combined_probs.sum()
        else:
            combined_probs = self._emergency_fallback(guessed_letters)
        
        return combined_probs
    
    def _frequency_weighted_probabilities(self, possible_words, guessed_letters):
        """Calculate probabilities weighted by word frequencies"""
        letter_weights = np.zeros(26)
        total_weight = sum(freq for _, freq in possible_words)
        
        for word, frequency in possible_words:
            weight = frequency / total_weight
            for letter in set(word):
                if letter not in guessed_letters:
                    idx = ord(letter) - ord('A')
                    letter_weights[idx] += weight
        
        return letter_weights
    
    def _position_based_probabilities(self, pattern, guessed_letters):
        """Enhanced position-based probabilities"""
        probs = np.zeros(26)
        
        for pos, char in enumerate(pattern):
            if char == '_':  # Blank position
                if pos < len(self.position_frequencies):
                    # Use pre-calculated position frequencies
                    probs += self.position_frequencies[pos] * 2.0
        
        return probs
    
    def _bigram_context_probabilities(self, pattern, guessed_letters):
        """Use bigram context for smarter guessing"""
        probs = np.zeros(26)
        
        # Check left context
        for i in range(1, len(pattern)):
            if pattern[i] == '_' and pattern[i-1] != '_':
                left_char = pattern[i-1]
                if left_char in self.bigram_frequencies:
                    for right_char, count in self.bigram_frequencies[left_char].items():
                        if right_char not in guessed_letters:
                            idx = ord(right_char) - ord('A')
                            probs[idx] += count * 0.1
        
        # Check right context
        for i in range(len(pattern) - 1):
            if pattern[i] == '_' and pattern[i+1] != '_':
                right_char = pattern[i+1]
                for left_char, bigram_dict in self.bigram_frequencies.items():
                    if right_char in bigram_dict and left_char not in guessed_letters:
                        idx = ord(left_char) - ord('A')
                        probs[idx] += bigram_dict[right_char] * 0.1
        
        return probs
    
    def _game_state_weights(self, wrong_guesses, num_possible_words):
        """Dynamic weights based on game state"""
        remaining_guesses = 6 - wrong_guesses
        
        if remaining_guesses <= 2:
            # Emergency mode - prioritize frequency
            return {'frequency': 3.0, 'position': 1.0, 'bigram': 0.5}
        elif num_possible_words <= 5:
            # Few possibilities - be precise
            return {'frequency': 1.0, 'position': 2.0, 'bigram': 1.5}
        else:
            # Normal mode - balanced approach
            return {'frequency': 1.5, 'position': 1.0, 'bigram': 1.0}
    
    def _vowel_consonant_strategy(self, probabilities, guessed_letters, wrong_guesses):
        """Strategic vowel/consonant guessing"""
        guessed_vowels = [l for l in guessed_letters if l in 'AEIOU']
        guessed_consonants = [l for l in guessed_letters if l not in 'AEIOU']
        
        # If few vowels guessed and many blanks, prioritize vowels
        if len(guessed_vowels) < 3 and wrong_guesses < 4:
            for vowel in 'AEIOU':
                if vowel not in guessed_letters:
                    idx = ord(vowel) - ord('A')
                    probabilities[idx] *= 1.3
        
        return probabilities
    
    def _advanced_fallback(self, guessed_letters, pattern, wrong_guesses):
        """Sophisticated fallback strategy"""
        probs = np.zeros(26)
        unguessed = [c for c in string.ascii_uppercase if c not in guessed_letters]
        
        if not unguessed:
            return probs
        
        # Use multiple fallback strategies
        for letter in unguessed:
            idx = ord(letter) - ord('A')
            
            # Base frequency
            base_freq = self.letter_frequencies.get(letter, 0.01)
            
            # Position bonus for common positions
            pos_bonus = 0
            for i, char in enumerate(pattern):
                if char == '_' and i < len(self.position_frequencies):
                    pos_bonus += self.position_frequencies[i, idx]
            
            # Vowel/consonant strategy
            strategy_bonus = 1.0
            if letter in 'AEIOU' and wrong_guesses < 4:
                strategy_bonus = 1.5  # Prioritize vowels early
            
            probs[idx] = base_freq * (1 + pos_bonus) * strategy_bonus
        
        # Normalize
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            for letter in unguessed:
                idx = ord(letter) - ord('A')
                probs[idx] = 1.0
            probs /= probs.sum()
        
        return probs
    
    def _emergency_fallback(self, guessed_letters):
        """Final emergency fallback"""
        probs = np.zeros(26)
        unguessed = [c for c in string.ascii_uppercase if c not in guessed_letters]
        
        # Most common English letters in order
        common_order = ['E', 'T', 'A', 'O', 'I', 'N', 'S', 'R', 'H', 'L', 'D', 'C', 'U', 'M', 'W', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', 'J', 'X', 'Q', 'Z']
        
        for i, letter in enumerate(common_order):
            if letter in unguessed:
                idx = ord(letter) - ord('A')
                probs[idx] = len(common_order) - i  # Higher weight for more common letters
        
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            for letter in unguessed:
                idx = ord(letter) - ord('A')
                probs[idx] = 1.0
            probs /= probs.sum()
        
        return probs

# ==================== ENHANCED GAME ENGINE ====================

class EnhancedHangmanGame:
    def __init__(self, ai_model, word_list):
        self.ai = ai_model
        self.word_list = word_list
        self.reset()
    
    def reset(self, target_word=None):
        if target_word:
            self.target_word = target_word.upper()
        else:
            self.target_word = random.choice(self.word_list).upper()
        
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.max_wrong_guesses = 6
        self.correct_guesses = 0
        self.game_over = False
        self.won = False
        self.guess_history = []
        self.repeated_guesses = 0
        
        return self.get_game_state()
    
    def get_game_state(self):
        masked = ''.join(c if c in self.guessed_letters else '_' for c in self.target_word)
        return {
            'masked_word': masked,
            'target_word': self.target_word,
            'guessed_letters': self.guessed_letters.copy(),
            'wrong_guesses': self.wrong_guesses,
            'remaining_guesses': self.max_wrong_guesses - self.wrong_guesses,
            'correct_guesses': self.correct_guesses,
            'game_over': self.game_over,
            'won': self.won,
            'guess_history': self.guess_history.copy(),
            'repeated_guesses': self.repeated_guesses
        }
    
    def make_guess(self, letter):
        letter = letter.upper()
        result = {
            'letter': letter,
            'is_repeated': False,
            'is_correct': False,
            'positions_found': []
        }
        
        if self.game_over:
            return self.get_game_state(), result
        
        # Check for repeated guess
        if letter in self.guessed_letters:
            result['is_repeated'] = True
            self.repeated_guesses += 1
            self.guess_history.append(letter)
            return self.get_game_state(), result
        
        self.guessed_letters.add(letter)
        self.guess_history.append(letter)
        
        # Check if correct
        if letter in self.target_word:
            result['is_correct'] = True
            for i, char in enumerate(self.target_word):
                if char == letter:
                    result['positions_found'].append(i)
                    self.correct_guesses += 1
        else:
            result['is_correct'] = False
            self.wrong_guesses += 1
        
        # Check game end
        if self.correct_guesses == len(self.target_word):
            self.game_over = True
            self.won = True
        elif self.wrong_guesses >= self.max_wrong_guesses:
            self.game_over = True
            self.won = False
        
        return self.get_game_state(), result
    
    def ai_make_guess(self):
        if self.game_over:
            return None, None, None
        
        state = self.get_game_state()
        
        # Get AI probabilities
        probabilities = self.ai.calculate_letter_probabilities(
            state['masked_word'], 
            state['guessed_letters'],
            state['wrong_guesses']
        )
        
        # Choose best letter with some exploration for learning
        best_letter = None
        best_prob = -1
        
        for i, prob in enumerate(probabilities):
            letter = chr(ord('A') + i)
            if letter not in self.guessed_letters and prob > best_prob:
                best_prob = prob
                best_letter = letter
        
        if best_letter is None:
            unguessed = [c for c in string.ascii_uppercase if c not in self.guessed_letters]
            best_letter = random.choice(unguessed) if unguessed else 'A'
        
        # Make the guess
        new_state, result = self.make_guess(best_letter)
        
        return best_letter, result, probabilities

# ==================== COMPETITION SCORING ====================

class CompetitionScorer:
    @staticmethod
    def calculate_final_score(results, num_games=2000):
        total_wins = sum(1 for result in results if result['won'])
        total_wrong_guesses = sum(result['wrong_guesses'] for result in results)
        total_repeated_guesses = sum(result['repeated_guesses'] for result in results)
        
        success_rate = total_wins / num_games
        
        final_score = (success_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
        
        return {
            'success_rate': success_rate,
            'total_wins': total_wins,
            'total_wrong_guesses': total_wrong_guesses,
            'total_repeated_guesses': total_repeated_guesses,
            'final_score': final_score
        }

# ==================== HIGH-PERFORMANCE EVALUATION ====================

class PerformanceEvaluator:
    def __init__(self):
        self.scorer = CompetitionScorer()
    
    def evaluate_ai_performance(self, ai, word_list, num_games=2000, verbose=True):
        """Comprehensive performance evaluation"""
        if verbose:
            print(f"\n🏆 RUNNING COMPREHENSIVE EVALUATION ({num_games} games)")
            print("=" * 60)
        
        results = []
        game_stats = []
        
        for game_num in range(num_games):
            game = EnhancedHangmanGame(ai, word_list)
            state = game.reset()
            
            # AI plays automatically
            while not state['game_over']:
                game.ai_make_guess()
                state = game.get_game_state()
            
            results.append({
                'won': state['won'],
                'wrong_guesses': state['wrong_guesses'],
                'repeated_guesses': state['repeated_guesses'],
                'total_guesses': len(state['guess_history']),
                'word': state['target_word']
            })
            
            game_stats.append(state)
            
            if verbose and (game_num + 1) % 500 == 0:
                print(f"  Completed {game_num + 1}/{num_games} games...")
        
        # Calculate scores
        score_data = self.scorer.calculate_final_score(results, num_games)
        
        if verbose:
            self._display_comprehensive_results(score_data, results)
        
        return score_data, results
    
    def _display_comprehensive_results(self, score_data, results):
        """Display detailed performance analysis"""
        print("\n" + "📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Basic metrics
        wins = score_data['total_wins']
        total_games = len(results)
        success_rate = score_data['success_rate']
        
        print(f"🎯 Success Rate: {success_rate:.1%} ({wins}/{total_games})")
        print(f"💔 Total Wrong Guesses: {score_data['total_wrong_guesses']}")
        print(f"🔄 Total Repeated Guesses: {score_data['total_repeated_guesses']}")
        
        # Advanced metrics
        avg_wrong = score_data['total_wrong_guesses'] / total_games
        avg_repeated = score_data['total_repeated_guesses'] / total_games
        avg_guesses = sum(r['total_guesses'] for r in results) / total_games
        
        print(f"\n📈 AVERAGE METRICS PER GAME:")
        print(f"   Wrong Guesses: {avg_wrong:.2f}")
        print(f"   Repeated Guesses: {avg_repeated:.2f}")
        print(f"   Total Guesses: {avg_guesses:.2f}")
        
        # Score breakdown
        print(f"\n🧮 COMPETITION SCORE BREAKDOWN:")
        print(f"   Success Component: {success_rate:.3f} × 2000 = {success_rate * 2000:.1f}")
        print(f"   Wrong Guesses Penalty: {score_data['total_wrong_guesses']} × 5 = -{score_data['total_wrong_guesses'] * 5}")
        print(f"   Repeated Guesses Penalty: {score_data['total_repeated_guesses']} × 2 = -{score_data['total_repeated_guesses'] * 2}")
        
        print(f"\n🏆 FINAL COMPETITION SCORE: {score_data['final_score']:.2f}")
        
        # Performance assessment
        self._assess_performance(score_data)
    
    def _assess_performance(self, score_data):
        """Assess and display performance level"""
        success_rate = score_data['success_rate']
        
        print(f"\n💡 PERFORMANCE ASSESSMENT:")
        print("-" * 30)
        
        if success_rate >= 0.75:
            print("🎉 OUTSTANDING! Performance exceeds 75% success rate!")
            print("   The AI demonstrates expert-level word guessing capability.")
        elif success_rate >= 0.65:
            print("👍 EXCELLENT! Performance between 65-75% success rate!")
            print("   The AI shows strong pattern recognition skills.")
        elif success_rate >= 0.55:
            print("✅ GOOD! Performance between 55-65% success rate!")
            print("   The AI is effective but has room for improvement.")
        else:
            print("💪 SOLID! Performance shows good foundation.")
            print("   Further optimization can improve results.")

# ==================== ENHANCED PRESENTATION SYSTEM ====================

class HighPerformanceHangmanDemo:
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file
        self.ai = AdvancedHangmanAI()
        self.evaluator = PerformanceEvaluator()
        self.load_corpus()
        
    def load_corpus(self):
        """Load and prepare corpus"""
        try:
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                corpus = [line.strip() for line in f if line.strip()]
            print(f"📚 Loaded {len(corpus)} words from corpus")
            self.ai.train(corpus)
            self.corpus = [word.upper() for word in corpus if 3 <= len(word) <= 12]
        except FileNotFoundError:
            print("❌ Corpus file not found. Using optimized sample words.")
            sample_words = [
                "PROGRAMMING", "COMPUTER", "ALGORITHM", "DATABASE", "SOFTWARE",
                "HARDWARE", "NETWORK", "SECURITY", "INTERNET", "SYSTEM",
                "APPLICATION", "DEVELOPMENT", "LANGUAGE", "PLATFORM", "FRAMEWORK",
                "HANGMAN", "PYTHON", "JAVA", "JAVASCRIPT", "HTML", "CSS", "SQL"
            ]
            self.ai.train(sample_words)
            self.corpus = sample_words
    
    def run_high_performance_benchmark(self, num_games=2000):
        """Run optimized performance benchmark"""
        print(f"\n🚀 RUNNING HIGH-PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        score_data, results = self.evaluator.evaluate_ai_performance(
            self.ai, self.corpus, num_games
        )
        
        # Additional performance insights
        self._show_performance_insights(score_data, results)
        
        return score_data, results
    
    def _show_performance_insights(self, score_data, results):
        """Show advanced performance insights"""
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        print("-" * 40)
        
        success_rate = score_data['success_rate']
        
        # Efficiency analysis
        avg_guesses = sum(r['total_guesses'] for r in results) / len(results)
        avg_wrong = score_data['total_wrong_guesses'] / len(results)
        
        print(f"⚡ EFFICIENCY METRICS:")
        print(f"   Average Guesses per Game: {avg_guesses:.2f}")
        print(f"   Average Wrong Guesses: {avg_wrong:.2f}")
        print(f"   Guess Efficiency: {(1 - avg_wrong/avg_guesses):.1%}")
        
        # Word length analysis
        word_lengths = [len(r['word']) for r in results]
        avg_length = sum(word_lengths) / len(word_lengths)
        print(f"   Average Word Length: {avg_length:.1f} letters")
        
        # Show some example games
        print(f"\n📝 SAMPLE GAME OUTCOMES:")
        wins = [r for r in results if r['won']]
        losses = [r for r in results if not r['won']]
        
        if wins:
            sample_win = wins[0]
            print(f"   ✅ Won '{sample_win['word']}' with {sample_win['wrong_guesses']} wrong guesses")
        if losses:
            sample_loss = losses[0]
            print(f"   ❌ Lost '{sample_loss['word']}' with {sample_loss['wrong_guesses']} wrong guesses")

    def demo_single_game(self, word=None):
        """Demo a single game with detailed output"""
        if not word:
            word = random.choice(self.corpus)
        
        print(f"\n🎮 DEMO GAME: AI vs '{word}'")
        print("=" * 40)
        
        game = EnhancedHangmanGame(self.ai, self.corpus)
        state = game.reset(word)
        
        round_num = 1
        while not state['game_over']:
            guess, result, probabilities = game.ai_make_guess()
            
            print(f"\nRound {round_num}:")
            print(f"  Current: {state['masked_word']}")
            print(f"  AI guesses: '{guess}'")
            
            if result['is_correct']:
                print(f"  ✅ Correct! Found at positions: {result['positions_found']}")
            else:
                print(f"  ❌ Wrong! Remaining guesses: {state['remaining_guesses']}")
            
            # Show top 3 next candidates
            if probabilities is not None:
                candidates = []
                for i, prob in enumerate(probabilities):
                    letter = chr(ord('A') + i)
                    if letter not in state['guessed_letters'] and prob > 0:
                        candidates.append((letter, prob))
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                if candidates:
                    top_3 = [f"{l}({p:.1%})" for l, p in candidates[:3]]
                    print(f"  Next candidates: {', '.join(top_3)}")
            
            state = game.get_game_state()
            round_num += 1
        
        print(f"\n🎯 FINAL RESULT: {'WIN' if state['won'] else 'LOSS'}")
        print(f"   Word: {state['target_word']}")
        print(f"   Total Guesses: {len(state['guess_history'])}")
        print(f"   Wrong Guesses: {state['wrong_guesses']}")
        print(f"   Repeated Guesses: {state['repeated_guesses']}")

# ==================== QUICK PERFORMANCE TEST ====================

def quick_performance_test():
    """Quick test to verify improved performance"""
    print("🚀 QUICK PERFORMANCE VERIFICATION")
    print("=" * 50)
    
    demo = HighPerformanceHangmanDemo("corpus.txt")
    
    # Run smaller test for quick verification
    print("\nRunning quick 100-game test...")
    score_data, _ = demo.run_high_performance_benchmark(100)
    
    print(f"\n🎯 QUICK RESULTS:")
    print(f"   Success Rate: {score_data['success_rate']:.1%}")
    print(f"   Final Score: {score_data['final_score']:.2f}")
    
    if score_data['success_rate'] >= 0.65:
        print("✅ SUCCESS: Performance target achieved!")
    else:
        print("⚠️  Working towards target...")
    
    # Demo a single game
    print(f"\n" + "="*50)
    demo.demo_single_game()

def full_competition_test():
    """Run full 2000-game competition test"""
    print("🏆 FULL COMPETITION EVALUATION (2000 games)")
    print("=" * 60)
    
    demo = HighPerformanceHangmanDemo("corpus.txt")
    score_data, _ = demo.run_high_performance_benchmark(2000)
    
    print(f"\n⭐ FINAL COMPETITION RESULTS:")
    print(f"   Success Rate: {score_data['success_rate']:.1%}")
    print(f"   Final Score: {score_data['final_score']:.2f}")
    
    # Project performance
    if score_data['success_rate'] >= 0.7:
        print("🎉 EXCELLENT! AI is competition-ready!")
    elif score_data['success_rate'] >= 0.6:
        print("👍 GREAT! Strong performance achieved!")
    else:
        print("💪 GOOD! Solid foundation for improvement.")

if __name__ == "__main__":
    # Run quick test first
    quick_performance_test()
    
    # Ask if user wants full test
    response = input("\nRun full 2000-game competition test? (y/n): ")
    if response.lower() == 'y':
        full_competition_test()