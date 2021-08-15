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
 * A representation of the literals of the enumeration '<em><b>unarop</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see promela.PromelaPackage#getunarop()
 * @model
 * @generated
 */
public enum unarop implements Enumerator {
	/**
	 * The '<em><b>NOT</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NOT_VALUE
	 * @generated
	 * @ordered
	 */
	NOT(0, "NOT", "NOT"),

	/**
	 * The '<em><b>NEG</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NEG_VALUE
	 * @generated
	 * @ordered
	 */
	NEG(1, "NEG", "NEG"),

	/**
	 * The '<em><b>NOT1</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NOT1_VALUE
	 * @generated
	 * @ordered
	 */
	NOT1(2, "NOT1", "NOT1");

	/**
	 * The '<em><b>NOT</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>NOT</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NOT
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int NOT_VALUE = 0;

	/**
	 * The '<em><b>NEG</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>NEG</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NEG
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int NEG_VALUE = 1;

	/**
	 * The '<em><b>NOT1</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>NOT1</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NOT1
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int NOT1_VALUE = 2;

	/**
	 * An array of all the '<em><b>unarop</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final unarop[] VALUES_ARRAY =
		new unarop[] {
			NOT,
			NEG,
			NOT1,
		};

	/**
	 * A public read-only list of all the '<em><b>unarop</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<unarop> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>unarop</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static unarop get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			unarop result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>unarop</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static unarop getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			unarop result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>unarop</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static unarop get(int value) {
		switch (value) {
			case NOT_VALUE: return NOT;
			case NEG_VALUE: return NEG;
			case NOT1_VALUE: return NOT1;
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
	private unarop(int value, String name, String literal) {
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
	
} //unarop
