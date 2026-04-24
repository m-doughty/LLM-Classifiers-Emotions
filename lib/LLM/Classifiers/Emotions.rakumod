use ONNX::Native;
use ONNX::Native::Types;
use Tokenizers;
use JSON::Fast;

unit module LLM::Classifiers::Emotions;

# === Exceptions ===

class X::LLM::Classifiers::Emotions::ModelMissing is Exception is export {
	has Str $.searched;

	method message(--> Str) {
		"[LLM::Classifiers::Emotions] Model not staged at { $!searched }. "
		~ "Reinstall via `zef install LLM::Classifiers::Emotions --force-install`, "
		~ "or set LLM_EMOTIONS_MODEL_DIR to a directory containing tokenizer.json, "
		~ "config.json, and model.onnx."
	}
}

class X::LLM::Classifiers::Emotions::InvalidConfig is Exception is export {
	has Str $.reason;

	method message(--> Str) {
		"[LLM::Classifiers::Emotions] Invalid config.json: $!reason";
	}
}

# === Model directory resolution ===
#
# Mirrors ONNX::Native::FFI's three-tier pattern:
#
#   1. $LLM_EMOTIONS_MODEL_DIR env var — explicit override.
#   2. XDG-staged dir keyed by BINARY_TAG (the install-time path).
#   3. Fail loudly with a pointer at the reinstall path.
#
# No bare-name fallback — unlike libraries, model files don't have
# a system loader to defer to.

sub _staged-dir(--> IO::Path) {
	my $res = %?RESOURCES<BINARY_TAG>;
	my Str $tag = '';
	if $res.defined && $res.IO.f {
		$tag = (try $res.IO.slurp.trim) // '';
	}
	return IO::Path unless $tag.chars;
	my Str $base = %*ENV<LLM_EMOTIONS_DATA_DIR>
		// %*ENV<XDG_DATA_HOME>
		// ($*DISTRO.is-win
				?? (%*ENV<LOCALAPPDATA>
						// "{%*ENV<USERPROFILE> // '.'}\\AppData\\Local")
				!! "{%*ENV<HOME> // '.'}/.local/share");
	"$base/LLM-Classifiers-Emotions/$tag".IO;
}

sub _resolve-model-dir(IO::Path $override --> IO::Path) {
	with $override {
		return $override if $override.d;
		X::LLM::Classifiers::Emotions::ModelMissing.new(
			searched => $override.Str).throw;
	}
	with %*ENV<LLM_EMOTIONS_MODEL_DIR> -> $env-dir {
		return $env-dir.IO if $env-dir.IO.d;
		X::LLM::Classifiers::Emotions::ModelMissing.new(
			searched => $env-dir).throw;
	}
	my $staged = _staged-dir();
	if $staged.defined && $staged.d {
		return $staged;
	}
	X::LLM::Classifiers::Emotions::ModelMissing.new(
		searched => ($staged.defined ?? $staged.Str !! '(BINARY_TAG missing)')
	).throw;
}

# === Classifier ===

class Classifier is export {
	has ONNX::Native::Session $!session;
	has Tokenizers            $!tokenizer;
	has Str                   @!id2label;
	has IO::Path              $!model-dir;
	has Bool                  $!disposed;

	submethod BUILD(
		:$!session!, :$!tokenizer!, :@!id2label!,
		:$!model-dir!,
	) {
		$!disposed = False;
	}

	#| Load the staged model + tokenizer + label map. Construct
	#| once per process; classifier reuse is cheap after that.
	method new(
		:@providers = (CPU,),
		IO::Path :$model-dir = IO::Path,
		Str :$log-id = 'llm-emotions',
		--> Classifier:D
	) {
		my IO::Path $dir = _resolve-model-dir($model-dir);
		my IO::Path $tokenizer-path = $dir.add('tokenizer.json');
		my IO::Path $config-path    = $dir.add('config.json');
		my IO::Path $model-path     = $dir.add('model.onnx');

		for $tokenizer-path, $config-path, $model-path -> $f {
			unless $f.e {
				X::LLM::Classifiers::Emotions::ModelMissing.new(
					searched => $f.Str).throw;
			}
		}

		my @id2label = self!load-id2label($config-path);
		my $tokenizer = Tokenizers.new-from-json($tokenizer-path.slurp);
		my $session = ONNX::Native::Session.new(
			:path($model-path.Str),
			:@providers,
			:$log-id,
		);

		self.bless(
			:$session, :$tokenizer, :@id2label, :model-dir($dir),
		);
	}

	method !load-id2label(IO::Path $config-path --> Array[Str]) {
		my %cfg = try { from-json($config-path.slurp) };
		without %cfg {
			X::LLM::Classifiers::Emotions::InvalidConfig.new(
				reason => "Could not parse JSON at $config-path").throw;
		}
		my $id2label = %cfg<id2label>;
		without $id2label {
			X::LLM::Classifiers::Emotions::InvalidConfig.new(
				reason => "config.json has no id2label field").throw;
		}
		# HuggingFace writes id2label as { "0": "anger", "1": ... };
		# we want a dense Str array indexed by the integer id.
		my Int $max = $id2label.keys.map(*.Int).max;
		my Str @out;
		@out[$_] = $id2label{$_.Str} // '' for 0..$max;
		@out;
	}

	#| Current model directory — useful for debugging / logging.
	method model-dir(--> IO::Path) { $!model-dir }

	#| List of all label names in id order (28 elements for
	#| Cohee's go-emotions model).
	method labels(--> List) {
		self!ensure-live;
		@!id2label.List;
	}

	#| Classify one text. Returns a list of Hash, one per label,
	#| sorted descending by score. Scores sum to 1.0 ± 1e-5
	#| (softmax invariant). Example:
	#|     ({ label => 'joy', score => 0.89 },
	#|      { label => 'love', score => 0.04 }, ...)
	method classify(Str:D $text --> List) {
		my @logits = self.raw-logits($text);
		my @probs = softmax(@logits);
		my @pairs;
		for ^@probs.elems -> $i {
			@pairs.push: %(
				label => @!id2label[$i],
				score => @probs[$i],
			);
		}
		@pairs.sort({ -$^a<score> }).List;
	}

	#| Argmax label — the most-likely single emotion. Returns the
	#| Str type object (undefined) when :$min-score is set and no
	#| label meets it.
	method top(Str:D $text, Numeric :$min-score --> Str) {
		my @results = self.classify($text);
		my %top = @results[0];
		with $min-score {
			return Str if %top<score> < $min-score;
		}
		%top<label>;
	}

	#| Run the ONNX session and return the raw 28-element logits
	#| list (pre-softmax). Exposed publicly so tests can assert on
	#| bit-exact values for determinism checks — the softmax step
	#| is cheap but can hide sub-LSB differences in the raw
	#| output.
	# DistilBERT's positional embeddings are fixed at 512. Cohee's
	# tokenizer.json has `truncation: null`, so we truncate here
	# rather than rely on the tokenizer — feeding a longer
	# sequence to the session crashes inside ORT with a
	# broadcast-dimension mismatch in the embeddings Add node.
	constant MAX-SEQ-LEN = 512;

	method raw-logits(Str:D $text --> List) {
		self!ensure-live;
		my @ids = $!tokenizer.encode($text, :add-special-tokens);
		@ids = @ids[^MAX-SEQ-LEN] if @ids.elems > MAX-SEQ-LEN;
		my @mask = (1) xx @ids.elems;

		my $input-ids = ONNX::Native::Tensor.from-ints(
			@ids, :shape([1, @ids.elems]), :dtype(INT64));
		my $attention = ONNX::Native::Tensor.from-ints(
			@mask, :shape([1, @mask.elems]), :dtype(INT64));

		my %out = $!session.run(
			inputs  => { input_ids => $input-ids,
				     attention_mask => $attention },
			outputs => ['logits',],
		);

		my @logits = %out<logits>.to-num-array;
		$input-ids.dispose;
		$attention.dispose;
		.dispose for %out.values;
		@logits.List;
	}

	method dispose(--> Nil) {
		return if $!disposed;
		$!disposed = True;
		.dispose with $!session;
		$!session = Nil;
	}

	submethod DESTROY() {
		self.dispose;
	}

	method !ensure-live() {
		if $!disposed || !$!session.defined {
			die "LLM::Classifiers::Emotions::Classifier has been disposed";
		}
	}
}

# === Math ===
#
# Numerically stable softmax — subtract max first, exp, divide by
# sum. Prevents overflow on extreme logits.
sub softmax(@logits --> List) is export {
	return () unless @logits.elems;
	my Num $max = @logits.max.Num;
	my @exps = @logits.map({ ($_ - $max).Num.exp });
	my Num $sum = @exps.sum.Num;
	return (0e0 xx @logits.elems).list if $sum == 0;
	@exps.map({ $_ / $sum }).list;
}
