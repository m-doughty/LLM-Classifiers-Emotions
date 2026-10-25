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
#| Licence notices. Whichever path staged the model, the staged
#| dir also ends up holding the notices for it: THIRD-PARTY.md
#| (each upstream component, its licence, source and credit),
#| MODEL-CARD.md (the model's README.md at the pinned revision)
#| and LICENSES/ (the MIT and Apache-2.0 texts). The release
#| tarball carries all three, because the release redistributes
#| the model and those are the notices its licences require;
#| Build stages them as shipped. Anything missing — the HF
#| fallback, a tarball without them, or a dir staged by an
#| older Build — is written from the dist's resources
#| (licenses/ and THIRD-PARTY.md.in, the same template
#| hf-mirror.yml renders) and the model card is fetched from
#| HF at the pinned revision. The card is best-effort: a
#| failed fetch warns and the install carries on, since
#| THIRD-PARTY.md names the card's URL. Notices are never part
#| of the model files' checksum verification.
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
#|                                   github.com/m-doughty/
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

	# Files downloaded from HF in the fallback path. Order:
	# (staged-basename, hf-path).
	constant @HF-FILES = (
		('tokenizer.json',  'tokenizer.json'),
		('config.json',     'config.json'),
		('model.onnx',      'onnx/model_quantized.onnx'),
	);

	# The licence notices staged beside the model files, and where
	# each comes from: the model card is the HF repo's README.md
	# at the pinned revision; THIRD-PARTY.md is rendered from the
	# template in resources/; LICENSES/ is a copy of every text in
	# resources/licenses/.
	constant $CARD-NAME        = 'MODEL-CARD.md';
	constant $CARD-HF-PATH     = 'README.md';
	constant $THIRD-PARTY-NAME = 'THIRD-PARTY.md';
	constant $TEMPLATE-NAME    = 'THIRD-PARTY.md.in';
	constant $LICENSES-NAME    = 'LICENSES';

	#| A HuggingFace::API client, or anything else with its
	#| `get-file-to-file($model-id, $hf-path, IO::Path $out,
	#| :$revision)` method. zef constructs Build with no
	#| arguments, so in a real install this starts undefined and
	#| the HF paths require HuggingFace::API on first use; the
	#| tests pass an offline stand-in.
	has $.hf-api;

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
			# A dir staged by an older Build has the model but no
			# notices; top them up rather than re-download 70 MB.
			self.stage-notices($dist-path, $stage, :$revision, :$binary-tag);
			say "✅ LLM::Classifiers::Emotions model already staged at $stage.";
			return True;
		}

		unless $from-hf {
			my Bool $ok = self!try-primary(
				$dist-path, $binary-tag, $stage);
			if $ok {
				self.stage-notices($dist-path, $stage, :$revision, :$binary-tag);
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
		self.stage-notices($dist-path, $stage, :$revision, :$binary-tag);
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

		my Str $actual = self.sha256($cached);
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
				self!remove-tree($entry);
			}
		}
		$stage.mkdir;
		# The archive is named from its own directory: a GNU tar ahead
		# of Windows' own on the PATH (Git for Windows ships one) reads
		# the 'C:' of an absolute archive path as a remote host name.
		my $rc = run 'tar', '-xzf', $archive.basename, '-C', $stage.absolute,
			:cwd($archive.absolute.IO.parent);
		die "❌ Failed to extract $archive." unless $rc.exitcode == 0;
	}

	# --- Fallback: HuggingFace via HuggingFace::API ---------------------

	method !try-hf-fallback($dist-path, Str $revision, IO::Path $stage) {
		my $api = self!hf-client;

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

	method !hf-client() {
		return $!hf-api if $!hf-api.defined;
		# HuggingFace::API is listed in build-depends so it's
		# available by the time Build runs. We lazy-load it here
		# to avoid the dependency cost on the primary-only path.
		require HuggingFace::API;
		$!hf-api = ::('HuggingFace::API').new;
	}

	# --- Licence notices ------------------------------------------------

	#| Stage the licence notices into $stage, beside the model
	#| files: LICENSES/ (a copy of every text in the dist's
	#| resources/licenses/, dotfiles aside), THIRD-PARTY.md
	#| (rendered from resources/THIRD-PARTY.md.in) and
	#| MODEL-CARD.md (the HF repo's README.md at $revision).
	#|
	#| Only what is missing is written. Notices that arrived in
	#| the release tarball are what that release redistributed,
	#| so they are staged exactly as shipped and never replaced.
	#| Each file is written under a temporary name and renamed,
	#| so an interrupted install can't leave a truncated notice
	#| that a later run would then keep.
	#|
	#| Dies if the dist's own notice sources are missing or the
	#| template doesn't render — that is a broken distribution,
	#| not something to install around. The model card is the
	#| exception: it comes over the network, so a failed fetch
	#| or a checksum mismatch warns and leaves MODEL-CARD.md out.
	method stage-notices(IO() $dist-path, IO::Path:D $stage,
			Str:D :$revision!, Str:D :$binary-tag! --> Nil) {
		my IO::Path $resources = $dist-path.add('resources');

		my IO::Path $licenses-src = $resources.add('licenses');
		die "❌ Missing licence texts: $licenses-src is not a directory."
			unless $licenses-src.d;
		# Dotfiles (a Finder .DS_Store) are not licence texts; the
		# workflow's `resources/licenses/*` glob skips them too.
		my @texts = $licenses-src.dir
			.grep({ .f && !.basename.starts-with('.') })
			.sort(*.basename);
		die "❌ Missing licence texts: $licenses-src is empty."
			unless @texts;

		my IO::Path $licenses-out = $stage.add($LICENSES-NAME);
		$licenses-out.mkdir;
		for @texts -> IO::Path $text {
			my IO::Path $dst = $licenses-out.add($text.basename);
			next if $dst.e;
			self!write-via-partial($dst, $text.slurp(:bin));
		}

		my IO::Path $third-party = $stage.add($THIRD-PARTY-NAME);
		unless $third-party.e {
			my IO::Path $template = $resources.add($TEMPLATE-NAME);
			die "❌ Missing notice template: $template" unless $template.f;
			my Str $doc = self.render-third-party(
				$template.slurp, :$revision, :$binary-tag);
			self!write-via-partial($third-party, $doc.encode('utf8'));
		}

		my IO::Path $card = $stage.add($CARD-NAME);
		self!fetch-model-card($dist-path, $card, $revision)
			unless $card.e;
	}

	#| Render the THIRD-PARTY.md template: substitute the pinned
	#| revision for `@REVISION@` and the release tag for
	#| `@BINARY_TAG@`. hf-mirror.yml renders the same template
	#| with the same two substitutions, so the document in the
	#| release tarball and the one written here match.
	#|
	#| Dies on a revision that isn't a 40-hex commit id, a tag
	#| that isn't a `binaries-` tag, or any `@NAME@` placeholder
	#| the substitutions left behind.
	method render-third-party(Str:D $template, Str:D :$revision!,
			Str:D :$binary-tag! --> Str:D) {
		die "❌ Revision '$revision' is not a 40-character "
		  ~ "lower-case hex commit id."
			unless $revision ~~ / ^ <[0..9 a..f]> ** 40 $ /;
		die "❌ Binary tag '$binary-tag' is not a binaries-<…> tag "
		  ~ "of letters, digits, '.', '_' and '-'."
			unless $binary-tag ~~ / ^ 'binaries-' <[A..Z a..z 0..9 . _ -]>+ $ /;

		my Str $doc = $template.subst('@REVISION@', $revision, :g);
		$doc = $doc.subst('@BINARY_TAG@', $binary-tag, :g);
		if $doc ~~ / '@' <[A..Z _]>+ '@' / {
			die "❌ $TEMPLATE-NAME has an unknown placeholder '$/'.";
		}
		$doc;
	}

	method !fetch-model-card($dist-path, IO::Path:D $card, Str:D $revision --> Nil) {
		my IO::Path $partial = $card.sibling($card.basename ~ '.partial');
		{
			CATCH {
				default {
					my $err = $_;
					my Str $why = (try $err.message.Str) // $err.^name;
					$partial.unlink if $partial.e;
					note "⚠️  Could not fetch the model card ($CARD-HF-PATH "
					   ~ "at {$revision.substr(0, 8)}): $why. "
					   ~ "Staged without $CARD-NAME; $THIRD-PARTY-NAME "
					   ~ "names the card's URL.";
				}
			}
			say "⬇️  Fetching $CARD-HF-PATH (model card) from $MODEL-ID "
			  ~ "@ {$revision.substr(0, 8)}...";
			self!hf-client.get-file-to-file($MODEL-ID, $CARD-HF-PATH,
				$partial, :$revision);
			my Str $expected = self!expected-sha($dist-path, $CARD-NAME);
			with $expected {
				my Str $actual = self.sha256($partial);
				unless $actual.defined && $actual.lc eq $expected.lc {
					$partial.unlink;
					note "⚠️  Checksum mismatch for the model card "
					   ~ "(expected $expected, got {$actual // 'unknown'}). "
					   ~ "Staged without $CARD-NAME; $THIRD-PARTY-NAME "
					   ~ "names the card's URL.";
					return;
				}
			}
			else {
				note "No checksum recorded for $CARD-NAME — skipping verify.";
			}
			$partial.rename($card);
		}
	}

	method !write-via-partial(IO::Path:D $dst, Blob:D $bytes --> Nil) {
		my IO::Path $partial = $dst.sibling($dst.basename ~ '.partial');
		$partial.spurt($bytes);
		$partial.rename($dst);
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
			my Str $actual = self.sha256($file);
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

	#| SHA-256 of $file as lower-case hex, or Str if it couldn't
	#| be computed. `shasum` on POSIX, `certutil` on Windows.
	method sha256(IO::Path $file --> Str) {
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

	# Delete $path and, if it is a directory, everything under
	# it. A symlink is unlinked, never followed. Raku's own IO
	# rather than `rm -rf`, which Windows doesn't have.
	method !remove-tree(IO::Path:D $path --> Nil) {
		if $path.l || !$path.d {
			$path.unlink;
			return;
		}
		self!remove-tree($_) for $path.dir;
		$path.rmdir;
	}
}
