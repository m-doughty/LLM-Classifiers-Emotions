#| Build.rakumod for LLM::Classifiers::Emotions.
#|
#| Stages Cohee's go-emotions DistilBERT model (tokenizer +
#| config + quantized onnx weights) into an XDG data dir at
#| install time, so tests and runtime code can find it offline.
#|
#| Two paths, tried in order:
#|
#|   1. Primary: download `cohee-goemotions-<BINARY_TAG>.tar.gz`
#|      from the module's GitHub Release, verify the tarball's
#|      sha256 against resources/checksums.txt, extract, verify
#|      each file's sha256 inside.
#|   2. Fallback: via HuggingFace::API, pull tokenizer.json,
#|      config.json, and onnx/model_quantized.onnx directly from
#|      the pinned HF revision, verify per-file sha256 (same
#|      checksums.txt entries), stage.
#|
#| The fallback is important for dev iteration and for users
#| installing before the mirror CI has run — but the primary
#| path is what we want in production so that installs don't
#| depend on HuggingFace's availability.
#|
#| Why we don't use META6 resources for the model files: zef
#| hashes every resource filename at stage time, which makes
#| runtime discovery flaky (we'd have to walk %?RESOURCES rather
#| than rely on a known filename). XDG staging with a BINARY_TAG
#| sentinel file is the Vips-Native / Notcurses-Native / ONNX-Native
#| convention.
#|
#| Env-var knobs:
#|
#|   LLM_EMOTIONS_BINARY_URL=<url>   override GitHub Release base
#|                                   URL (default:
#|                                   github.com/invisietch/
#|                                   LLM-Classifiers-Emotions/
#|                                   releases/download)
#|   LLM_EMOTIONS_BINARY_ONLY=1      refuse the HF fallback; fail
#|                                   if the primary path doesn't
#|                                   complete (CI-friendly)
#|   LLM_EMOTIONS_FROM_HF=1          skip primary, go straight to
#|                                   the HF fallback (dev / debug)
#|   LLM_EMOTIONS_CACHE_DIR=<path>   override download cache dir
#|                                   (default $XDG_CACHE_HOME)
#|   LLM_EMOTIONS_DATA_DIR=<path>    override staged base dir
#|                                   (default $XDG_DATA_HOME)
#|   LLM_EMOTIONS_MODEL_DIR=<path>   (runtime) load model from
#|                                   this dir, bypassing the
#|                                   staged-lookup entirely.

class Build {

	# --- Constants ------------------------------------------------------

	constant $DEFAULT-BASE-URL =
		'https://github.com/m-doughty/LLM-Classifiers-Emotions/releases/download';

	constant $MODEL-ID = 'Cohee/distilbert-base-uncased-go-emotions-onnx';

	#| Files downloaded from HF in the fallback path. Order:
	#| (staged-basename, hf-path).
	constant @HF-FILES = (
		('tokenizer.json',  'tokenizer.json'),
		('config.json',     'config.json'),
		('model.onnx',      'onnx/model_quantized.onnx'),
	);

	# --- Entry point ----------------------------------------------------

	method build($dist-path) {
		my Bool $from-hf     = ?%*ENV<LLM_EMOTIONS_FROM_HF>;
		my Bool $binary-only = ?%*ENV<LLM_EMOTIONS_BINARY_ONLY>;

		my Str $binary-tag = self!read-trimmed("$dist-path/BINARY_TAG");
		my Str $revision   = self!read-trimmed("$dist-path/resources/REVISION");

		# BINARY_TAG into resources so Emotions.rakumod can
		# locate the staged dir at runtime.
		self!stage-tag-file($dist-path, 'BINARY_TAG', $binary-tag);
		self!stage-tag-file($dist-path, 'REVISION',   $revision);

		my IO::Path $stage  = self!staged-dir($binary-tag);
		my IO::Path $marker = $stage.add('.staged-ok');

		if $marker.e && self!all-files-present($stage) {
			say "✅ LLM::Classifiers::Emotions model already staged at $stage.";
			return True;
		}

		unless $from-hf {
			my Bool $ok = self!try-primary(
				$dist-path, $binary-tag, $stage);
			if $ok {
				$marker.spurt("ok\n");
				say "✅ Staged go-emotions model via GitHub Release → $stage.";
				return True;
			}

			if $binary-only {
				die "❌ LLM_EMOTIONS_BINARY_ONLY=1 but primary "
				  ~ "path failed for $binary-tag.";
			}

			note "⚠️  Primary (GitHub Release) path failed — "
			   ~ "falling back to HuggingFace.";
		}

		self!try-hf-fallback($dist-path, $revision, $stage);
		$marker.spurt("ok\n");
		say "✅ Staged go-emotions model via HuggingFace → $stage.";
		True;
	}

	# --- Primary: GitHub Release tarball --------------------------------

	method !try-primary($dist-path, Str $binary-tag, IO::Path $stage --> Bool) {
		my Str $artifact = "cohee-goemotions-$binary-tag.tar.gz";
		my IO::Path $cache-dir = self!cache-dir($binary-tag);
		my IO::Path $cached = $cache-dir.add($artifact);
		my Str $base-url = %*ENV<LLM_EMOTIONS_BINARY_URL> // $DEFAULT-BASE-URL;
		my Str $url = "$base-url/$binary-tag/$artifact";

		unless $cached.e {
			$cache-dir.mkdir;
			say "⬇️  Fetching $artifact from $url";
			my $rc = run 'curl', '-fL', '--progress-bar',
				'-o', $cached.Str, $url;
			unless $rc.exitcode == 0 {
				$cached.unlink if $cached.e;
				return False;
			}
		}

		my Str $expected = self!expected-sha($dist-path, $artifact);
		without $expected {
			note "No checksum recorded for $artifact "
				~ "— refusing primary path.";
			return False;
		}

		my Str $actual = self!sha256($cached);
		unless $actual.defined && $actual.lc eq $expected.lc {
			note "Tarball checksum mismatch for $artifact "
				~ "(expected $expected, got {$actual // 'unknown'}).";
			$cached.unlink;
			return False;
		}

		self!extract-tarball($cached, $stage);
		self!verify-staged-files($dist-path, $stage);
		True;
	}

	method !extract-tarball(IO::Path $archive, IO::Path $stage) {
		if $stage.d {
			for $stage.dir -> $entry {
				next if $entry.basename eq '.staged-ok';
				$entry.d ?? run('rm', '-rf', $entry.Str)
					    !! $entry.unlink;
			}
		}
		$stage.mkdir;
		my $rc = run 'tar', '-xzf', $archive.Str, '-C', $stage.Str;
		die "❌ Failed to extract $archive." unless $rc.exitcode == 0;
	}

	# --- Fallback: HuggingFace via HuggingFace::API ---------------------

	method !try-hf-fallback($dist-path, Str $revision, IO::Path $stage) {
		# HuggingFace::API is listed in build-depends so it's
		# available by the time Build runs. We lazy-load it here
		# to avoid the dependency cost on the primary-only path.
		require HuggingFace::API;
		my $api = ::('HuggingFace::API').new;

		$stage.mkdir;

		for @HF-FILES -> ($basename, $hf-path) {
			my IO::Path $out = $stage.add($basename);
			say "⬇️  Fetching $hf-path from $MODEL-ID "
			  ~ "@ {$revision.substr(0, 8)}...";
			# get-file-to-file is the v0.2.0 addition.
			$api.get-file-to-file($MODEL-ID, $hf-path, $out,
				:$revision);
		}

		self!verify-staged-files($dist-path, $stage);
	}

	# --- Checksum plumbing ----------------------------------------------

	method !verify-staged-files($dist-path, IO::Path $stage) {
		for @HF-FILES -> ($basename, $) {
			my IO::Path $file = $stage.add($basename);
			unless $file.e {
				die "❌ Staged file missing: $file";
			}
			my Str $expected = self!expected-sha($dist-path, $basename);
			without $expected {
				note "No checksum recorded for $basename — skipping verify.";
				next;
			}
			my Str $actual = self!sha256($file);
			unless $actual.defined && $actual.lc eq $expected.lc {
				$file.unlink;
				die "❌ Checksum mismatch for $basename "
				  ~ "(expected $expected, got {$actual // 'unknown'}). "
				  ~ "File deleted; reinstall to retry.";
			}
		}
	}

	method !expected-sha($dist-path, Str $artifact --> Str) {
		my IO::Path $file = "$dist-path/resources/checksums.txt".IO;
		return Str unless $file.e;
		for $file.slurp.lines -> Str $line {
			my Str $trimmed = $line.trim;
			next if $trimmed eq '' || $trimmed.starts-with('#');
			my @parts = $trimmed.words;
			next unless @parts.elems >= 2;
			return @parts[0] if @parts[1] eq $artifact;
		}
		Str;
	}

	method !sha256(IO::Path $file --> Str) {
		if $*DISTRO.is-win {
			my $proc = run 'certutil', '-hashfile', $file.Str,
				'SHA256', :out, :err;
			my $out = $proc.out.slurp(:close);
			$proc.err.slurp(:close);
			for $out.lines -> Str $line {
				my Str $t = $line.subst(/\s+/, '', :g).lc;
				return $t if $t.chars == 64
					&& $t ~~ /^ <[0..9a..f]>+ $/;
			}
			return Str;
		}
		my $proc = run 'shasum', '-a', '256', $file.Str, :out, :err;
		my $out = $proc.out.slurp(:close);
		$proc.err.slurp(:close);
		$out.words.head;
	}

	# --- Path helpers ---------------------------------------------------

	method !staged-dir(Str $binary-tag --> IO::Path) {
		my Str $base = %*ENV<LLM_EMOTIONS_DATA_DIR>
			// %*ENV<XDG_DATA_HOME>
			// ($*DISTRO.is-win
					?? (%*ENV<LOCALAPPDATA>
							// "{%*ENV<USERPROFILE> // '.'}\\AppData\\Local")
					!! "{%*ENV<HOME> // '.'}/.local/share");
		"$base/LLM-Classifiers-Emotions/$binary-tag".IO;
	}

	method !cache-dir(Str $binary-tag --> IO::Path) {
		my Str $base = %*ENV<LLM_EMOTIONS_CACHE_DIR>
			// %*ENV<XDG_CACHE_HOME>
			// "{%*ENV<HOME> // '.'}/.cache";
		"$base/LLM-Classifiers-Emotions-binaries/$binary-tag".IO;
	}

	method !stage-tag-file($dist-path, Str $name, Str $value) {
		my IO::Path $dst = "$dist-path/resources/$name".IO;
		$dst.parent.mkdir;
		$dst.spurt("$value\n");
	}

	method !read-trimmed(Str $path --> Str) {
		my IO::Path $file = $path.IO;
		unless $file.e {
			die "❌ Missing required file: $path";
		}
		my Str $content = $file.slurp.trim;
		die "❌ File is empty: $path" unless $content.chars;
		$content;
	}

	method !all-files-present(IO::Path $stage --> Bool) {
		for @HF-FILES -> ($basename, $) {
			return False unless $stage.add($basename).e;
		}
		True;
	}
}
