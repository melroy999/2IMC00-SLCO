/**
 */
package slco;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.eclipse.emf.common.util.Enumerator;

/**
 * <!-- begin-user-doc -->
 * A representation of the literals of the enumeration '<em><b>Channel Type Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see slco.SlcoPackage#getChannelTypeEnum()
 * @model
 * @generated
 */
public enum ChannelTypeEnum implements Enumerator {
	/**
	 * The '<em><b>Async lossless</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ASYNC_LOSSLESS_VALUE
	 * @generated
	 * @ordered
	 */
	ASYNC_LOSSLESS(1, "async_lossless", "async_lossless"),

	/**
	 * The '<em><b>Async lossy</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ASYNC_LOSSY_VALUE
	 * @generated
	 * @ordered
	 */
	ASYNC_LOSSY(2, "async_lossy", "async_lossy"),

	/**
	 * The '<em><b>Sync</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SYNC_VALUE
	 * @generated
	 * @ordered
	 */
	SYNC(3, "sync", "sync");

	/**
	 * The '<em><b>Async lossless</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Async lossless</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ASYNC_LOSSLESS
	 * @model name="async_lossless"
	 * @generated
	 * @ordered
	 */
	public static final int ASYNC_LOSSLESS_VALUE = 1;

	/**
	 * The '<em><b>Async lossy</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Async lossy</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ASYNC_LOSSY
	 * @model name="async_lossy"
	 * @generated
	 * @ordered
	 */
	public static final int ASYNC_LOSSY_VALUE = 2;

	/**
	 * The '<em><b>Sync</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Sync</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SYNC
	 * @model name="sync"
	 * @generated
	 * @ordered
	 */
	public static final int SYNC_VALUE = 3;

	/**
	 * An array of all the '<em><b>Channel Type Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final ChannelTypeEnum[] VALUES_ARRAY =
		new ChannelTypeEnum[] {
			ASYNC_LOSSLESS,
			ASYNC_LOSSY,
			SYNC,
		};

	/**
	 * A public read-only list of all the '<em><b>Channel Type Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<ChannelTypeEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Channel Type Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static ChannelTypeEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			ChannelTypeEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Channel Type Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static ChannelTypeEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			ChannelTypeEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Channel Type Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static ChannelTypeEnum get(int value) {
		switch (value) {
			case ASYNC_LOSSLESS_VALUE: return ASYNC_LOSSLESS;
			case ASYNC_LOSSY_VALUE: return ASYNC_LOSSY;
			case SYNC_VALUE: return SYNC;
		}
		return null;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final int value;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final String name;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final String literal;

	/**
	 * Only this class can construct instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private ChannelTypeEnum(int value, String name, String literal) {
		this.value = value;
		this.name = name;
		this.literal = literal;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getValue() {
	  return value;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
	  return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getLiteral() {
	  return literal;
	}

	/**
	 * Returns the literal value of the enumerator, which is its string representation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		return literal;
	}
	
} //ChannelTypeEnum
