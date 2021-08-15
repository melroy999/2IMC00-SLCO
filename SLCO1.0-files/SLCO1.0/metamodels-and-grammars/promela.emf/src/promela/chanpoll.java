/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.eclipse.emf.common.util.Enumerator;

/**
 * <!-- begin-user-doc -->
 * A representation of the literals of the enumeration '<em><b>chanpoll</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see promela.PromelaPackage#getchanpoll()
 * @model
 * @generated
 */
public enum chanpoll implements Enumerator {
	/**
	 * The '<em><b>FULL</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #FULL_VALUE
	 * @generated
	 * @ordered
	 */
	FULL(0, "FULL", ""),

	/**
	 * The '<em><b>EMPTY</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #EMPTY_VALUE
	 * @generated
	 * @ordered
	 */
	EMPTY(1, "EMPTY", "EMPTY"),

	/**
	 * The '<em><b>NFULL</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NFULL_VALUE
	 * @generated
	 * @ordered
	 */
	NFULL(2, "NFULL", "NFULL"),

	/**
	 * The '<em><b>NEMTPY</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NEMTPY_VALUE
	 * @generated
	 * @ordered
	 */
	NEMTPY(3, "NEMTPY", "NEMTPY");

	/**
	 * The '<em><b>FULL</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>FULL</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #FULL
	 * @model literal=""
	 * @generated
	 * @ordered
	 */
	public static final int FULL_VALUE = 0;

	/**
	 * The '<em><b>EMPTY</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>EMPTY</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #EMPTY
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int EMPTY_VALUE = 1;

	/**
	 * The '<em><b>NFULL</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>NFULL</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NFULL
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int NFULL_VALUE = 2;

	/**
	 * The '<em><b>NEMTPY</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>NEMTPY</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NEMTPY
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int NEMTPY_VALUE = 3;

	/**
	 * An array of all the '<em><b>chanpoll</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final chanpoll[] VALUES_ARRAY =
		new chanpoll[] {
			FULL,
			EMPTY,
			NFULL,
			NEMTPY,
		};

	/**
	 * A public read-only list of all the '<em><b>chanpoll</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<chanpoll> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>chanpoll</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static chanpoll get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			chanpoll result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>chanpoll</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static chanpoll getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			chanpoll result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>chanpoll</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static chanpoll get(int value) {
		switch (value) {
			case FULL_VALUE: return FULL;
			case EMPTY_VALUE: return EMPTY;
			case NFULL_VALUE: return NFULL;
			case NEMTPY_VALUE: return NEMTPY;
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
	private chanpoll(int value, String name, String literal) {
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
	
} //chanpoll
